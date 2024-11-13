"""
An implementation of attention including several additional features and customizations over the standard pytorch implementation.
"""

import torch
from torch import nn
from einops import rearrange
import math
from attention_utils import repeat_kv, apply_rotary_emb, compute_causal_mask

class Attention(nn.Module):
    def __init__(self,
            d_model: int,
            n_heads: int,
            dropout: float,
            key_dim: int = None,
            n_kv_heads: int = None,
            add_bias_kv: bool = False,
            add_bias_out: bool = False,
            symmetric_attn: bool = False,
            total_n_heads: int = None):
        """
        An implementation of Attention with some added customization.

        Allows multi-query attention/grouped query attention, rotary positional embeddings,
        and custom relation activation functions.

        Parameters
        ----------
        d_model : int
            model dimension
        n_heads : int
            number of heads (query heads if n_kv_heads is set)
        dropout : float
            dropout rate
        n_kv_heads : int, optional
            number of key/value heads. used to implement multi-query attention or grouped query attention.
            n_kv_heads=1 corresponds to MQA, n_kv_heads > 1 corresponsd to grouped query attention.
            n_kv_heads=n_heads is standard MHA. uses MHA when None. By default None
        add_bias_kv : bool, optional
            whether to use bias in key/value projections, by default False
        add_bias_out : bool, optional
            whether to use bias in out projection, by default False
        symmetric_attn : bool, optional
            whether to weight-tie the query and key projections, making a symmetric attention criterion. By default False
        total_n_heads : int, optional
            total number of heads in dual attention (if using dual attention).
            used to ensure that concat(A, E) is of dimension d_model after concatentation.
            hence, output dimension is (d_model // total_heads) * n_heads.
            if None, total_heads = n_heads and output dimension is d_model
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads # number of heads (for query)
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads # n_kv_heads = 1 corresponds to multi-query attn
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.add_bias_out = add_bias_out
        self.symmetric_attn = symmetric_attn
        self.total_n_heads = n_heads if total_n_heads is None else total_n_heads # compatibility for dual attention

        self.key_dim = key_dim if key_dim is not None else self.d_model // self.total_n_heads # key dimension
        self.n_rep_kv = self.n_heads // self.n_kv_heads # use same kv heads for several query heads
        self.head_dim = self.d_model // self.total_n_heads # dim of projections
        assert self.n_heads % self.n_kv_heads == 0 # make sure n_kv_heads fits into n_heads (i.e., can be grouped)
        assert self.n_rep_kv * self.n_kv_heads == self.n_heads
        assert self.total_n_heads * self.head_dim == self.d_model

        self.attn_scale = 1 / math.sqrt(self.head_dim) # for scaled dot product attention

        self.wq = nn.Linear(self.d_model, self.n_heads * self.key_dim, bias=False)
        self.wk = nn.Linear(self.d_model, self.n_kv_heads * self.key_dim, bias=self.add_bias_kv)
        if symmetric_attn:
            self.wk = self.wq
        self.wv = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.n_heads * self.head_dim, bias=self.add_bias_out)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        freqs_cos: torch.Tensor = None,
        freqs_sin: torch.Tensor = None,
        attn_mask: torch.Tensor = None, # boolean attention mask: True indicates corresponding position *should* be attended to
        is_causal: bool = False, # indicates causal mask; should only set one of is_causal and attn_mask
        need_weights: bool = False
    ):
        """
        compute attention with given query, key, value.

        if freqs_cos and freqs_sin are given, apply rotary positional embeddings.
        if attn_mask is given, apply attention mask.
        if is_causal is True, apply causal mask (attn_mask must be None).

        Parameters
        ----------
        query : torch.Tensor
            query sequence of shape [bsz, len_in, d_model]
        key : torch.Tensor
            key sequence of shape [bsz, len_ctx, d_model]
        value : torch.Tensor
            value sequence of shape [bsz, len_ctx, d_model]
        freqs_cos : torch.Tensor, optional
            cosine of frequencies for RoPE. RoPE is applied if given.
            Note: RoPE does not work for cross-attention. By default None
        freqs_sin : torch.Tensor, optional
            cosine of frequencies for RoPE. RoPE is applied if given.
            Note: RoPE does not work for cross-attention. By default None
        attn_mask : torch.Tensor, optional
            boolean attention mask of shape [len_in, len_ctx]. True at [i,j] indicates i is allowed to attend to j.
            By default None
        is_causal : bool, optional
            whether to apply a causal mask. If True, attn_mask must be None. Only applies for self-attention.
            By default False
        need_weights : bool, optional
            whether to return the attention scores. If True, return value will be tuple (output, attn_scores).
            If True, will compute attention manually rather than using flash attention. By default False

        Returns
        -------
        torch.Tensor
            result of attention
        """

        bsz, qseqlen, _ = query.shape
        bsz, kseqlen, _ = key.shape
        bsz, vseqlen, _ = value.shape
        assert kseqlen == vseqlen

        # apply query/key/value projections and reshape to split into different heads
        xq, xk, xv = self.wq(query), self.wk(key), self.wv(value)
        xq = xq.view(bsz, qseqlen, self.n_heads, self.key_dim)
        xk = xk.view(bsz, kseqlen, self.n_kv_heads, self.key_dim)
        xv = xv.view(bsz, vseqlen, self.n_kv_heads, self.head_dim)

        # apply RoPE relative positional embeddings (if given)
        if freqs_cos is not None and freqs_sin is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        if self.n_rep_kv != 1:
            xk = repeat_kv(xk, self.n_rep_kv)  # (bs, seqlen, n_heads, key_dim)
            xv = repeat_kv(xv, self.n_rep_kv)  # (bs, seqlen, n_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, key_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # use flash implementation for softmax activation if weights not needed
        if not need_weights:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal, scale=self.attn_scale)
            scores = None

        # manual implementation for other kinds of activation
        else:
            # generate causal attn_mask
            assert not (attn_mask is not None and is_causal)
            if is_causal and attn_mask is None:
                attn_mask = compute_causal_mask(qseqlen, device=xq.device)
                # better to pass attn_mask rather than compute so that it doesn't need to be computed at each layer?

            # compute dot product
            scores = torch.matmul(xq, xk.transpose(2, 3)) * self.attn_scale

            # if softmax activation, masking is handled by adding -inf before softmax
            if attn_mask is not None:
                attn_mask_ = torch.zeros(qseqlen, kseqlen, dtype=xq.dtype, device=xq.device).masked_fill(attn_mask.logical_not(), float('-inf'))
                scores = scores + attn_mask_

            # apply softmax activation to inner products
            scores = torch.nn.functional.softmax(scores, dim=-1)

            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, qseqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output, scores
