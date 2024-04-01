"""
A module implementating relational cross-attention.
"""

import torch
from torch import nn
import model_utils
import math

from attention_utils import repeat_kv, apply_rotary_emb, compute_causal_mask


class RelationalCrossAttention(nn.Module):
    def __init__(self,
            d_model: int,
            n_heads: int,
            dropout: float,
            n_kv_heads: int = None,
            activation: str = 'softmax',
            add_bias_kv: bool = False,
            add_bias_out: bool = False,
            total_n_heads: int = None,
            use_relative_positional_symbols: bool = False):
        """
        An implementation of Relational Cross Attention with some added customization.

        Supports position-relative symbolic embeddings, multi-query attention/grouped query attention,
        and control over total number of heads (for use with "abstract attention").

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
        activation : str, optional
            name of activation function applied to attention scores. If softmax, flash attention is used.
            Otherwise, attention is computed 'manually' with the chosen activation function. By default 'softmax'.
        add_bias_kv : bool, optional
            whether to use bias in key/value projections, by default False
        add_bias_out : bool, optional
            whether to use bias in out projection, by default False
        total_n_heads : int, optional
            total number of heads in abstract attention (if using abstract attention).
            used to ensure that concat(A, E) is of dimension d_model after concatentation.
            hence, output dimension is (d_model // total_heads) * n_heads.
            if None, total_heads = n_heads and output dimension is d_model
        use_relative_pos_symbols : bool, optional
            whether to use relative positional symbols, by default False
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads # number of heads (for query)
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads # n_kv_heads = 1 corresponds to multi-query attn
        self.activation = activation # "relation activation function"
        self.activation_ = model_utils.get_activation_function(activation)
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.add_bias_out = add_bias_out
        self.total_n_heads = n_heads if total_n_heads is None else total_n_heads
        self.use_relative_positional_symbols = use_relative_positional_symbols

        self.n_rep_kv = self.n_heads // self.n_kv_heads # use same kv heads for several query heads
        self.head_dim = self.d_model // self.total_n_heads # dim of projections
        assert self.n_heads % self.n_kv_heads == 0 # make sure n_kv_heads fits into n_heads (i.e., can be grouped)
        assert self.n_rep_kv * self.n_kv_heads == self.n_heads
        assert self.total_n_heads * self.head_dim == self.d_model

        self.attn_scale = 1 / math.sqrt(self.head_dim) # for scaled dot product attention

        self.wq = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)
        self.wv = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.n_heads * self.head_dim, bias=self.add_bias_out)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        x: torch.Tensor,
        symbols: torch.Tensor,
        attn_mask: torch.Tensor = None, # boolean attention mask: True indicates corresponding position *should* be attended to
        is_causal: bool = False, # indicates causal mask; should only set one of is_causal and attn_mask
        ):
        """
        compute relational cross-attention attention with given input x and symbols.

        if attn_mask is given, apply attention mask.
        if is_causal is True, apply causal mask (attn_mask must be None).
        if use_relative_pos_symbols is True, the symbols are treated as relative positional embeddings.
            assumed to be of shape [len, len, dim] where len is the length of the sequence x.

        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape [bsz, len, d_model]
        symbols : torch.Tensor
            input tensor of shape [bsz, len, d_model] or [len, len, d_model] if use_relative_positional_symbols is True
        attn_mask : torch.Tensor, optional
            boolean attention mask of shape [len, len]. True at [i,j] indicates i is allowed to attend to j.
            By default None
        is_causal : bool, optional
            whether to apply a causal mask. If True, attn_mask must be None. By default False

        Returns
        -------
        torch.Tensor
            result of attention
        """

        bsz, seqlen, _ = x.shape

        # apply query/key/value projections and reshape to split into different heads
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(symbols)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)


        if self.use_relative_positional_symbols:
            # make sure symbols are of shape [len, len, dim]
            assert symbols.shape[0] == symbols.shape[1] == seqlen
            xv = xv.view(seqlen, seqlen, self.n_kv_heads, self.head_dim)
        else:
            xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # grouped multiquery attention: expand out keys and values
        if self.n_rep_kv != 1:
            xk = repeat_kv(xk, self.n_rep_kv)  # (bs, seqlen, n_heads, head_dim)
            xv = repeat_kv(xv, self.n_rep_kv)  # (bs, seqlen, n_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        # xv: (seq_len, seq_len, n_heads, head_dim) or (bs, seq_len, n_heads, head_dim)

        assert not (attn_mask is not None and is_causal)
        if is_causal and attn_mask is None:
            attn_mask = compute_causal_mask(seqlen, device=xq.device)
            # better to pass attn_mask rather than compute so that it doesn't need to be computed at each layer?

        # compute dot product
        scores = torch.matmul(xq, xk.transpose(2, 3)) * self.attn_scale # (bs, n_heads, seqlen, seqlen)

        # if softmax activation, masking is handled by adding -inf before softmax
        if attn_mask is not None and self.activation == 'softmax':
            attn_mask_ = torch.zeros(seqlen, seqlen, dtype=xq.dtype).masked_fill(attn_mask.logical_not(), float('-inf'))
            scores = scores + attn_mask_

        # apply (relation) activation to inner products
        scores = self.activation_(scores)

        # for non-softmax activation, masking is handled by zero-ing out *after* activation
        if attn_mask is not None and self.activation != 'softmax':
            scores = scores * attn_mask

        scores = self.attn_dropout(scores)

        if not self.use_relative_positional_symbols:
            xv = xv.transpose(1, 2)
            output = torch.matmul(scores, xv)  # (bs, n_heads, seqlen, head_dim)
        else:
            output = torch.einsum('bhij,ijhd->bihd', scores, xv)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output, scores

class DisentangledRelationalCrossAttention(nn.Module):
    def __init__(self,
            d_model: int,
            n_heads: int,
            dropout: float,
            n_kv_heads: int = None,
            rel_activation: str = 'identity',
            add_bias_kv: bool = False,
            add_bias_out: bool = False,
            total_n_heads: int = None,
            use_relative_positional_symbols: bool = False
            ):
        """
        An implementation of Disentangled Relational Cross Attention with some added customization.

        In Disentangled RCA, "attention" is separated from "relation". Two sets of projections are learned,
        one for attention and one for relational representation. Attention scores determine which objects to attend to,
        and relation scores compute the relation between the objects, which is tied to the symbols to identify to the "sender".

        Supports position-relative symbolic embeddings, multi-query attention/grouped query attention,
        and control over total number of heads (for use with "abstract attention").

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
        activation : str, optional
            name of activation function applied to attention scores. If softmax, flash attention is used.
            Otherwise, attention is computed 'manually' with the chosen activation function. By default 'softmax'.
        add_bias_kv : bool, optional
            whether to use bias in key/value projections, by default False
        add_bias_out : bool, optional
            whether to use bias in out projection, by default False
        total_n_heads : int, optional
            total number of heads in abstract attention (if using abstract attention).
            used to ensure that concat(A, E) is of dimension d_model after concatentation.
            hence, output dimension is (d_model // total_heads) * n_heads.
            if None, total_heads = n_heads and output dimension is d_model
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads # number of heads (for query)
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads # n_kv_heads = 1 corresponds to multi-query attn
        self.rel_activation = rel_activation # "relation activation function"
        self.rel_activation_ = model_utils.get_activation_function(rel_activation)
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.add_bias_out = add_bias_out
        self.total_n_heads = n_heads if total_n_heads is None else total_n_heads
        self.use_relative_positional_symbols = use_relative_positional_symbols

        self.n_rep_kv = self.n_heads // self.n_kv_heads # use same kv heads for several query heads
        self.head_dim = self.d_model // self.total_n_heads # dim of projections
        assert self.n_heads % self.n_kv_heads == 0 # make sure n_kv_heads fits into n_heads (i.e., can be grouped)
        assert self.n_rep_kv * self.n_kv_heads == self.n_heads
        assert self.total_n_heads * self.head_dim == self.d_model

        self.attn_scale = 1 / math.sqrt(self.head_dim) # for scaled dot product attention

        self.wq_attn = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk_attn = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)

        self.wq_rel = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk_rel = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)

        self.wv = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.n_heads * self.head_dim, bias=self.add_bias_out)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        x: torch.Tensor,
        symbols: torch.Tensor,
        attn_mask: torch.Tensor = None, # boolean attention mask: True indicates corresponding position *should* be attended to
        is_causal: bool = False # indicates causal mask; should only set one of is_causal and attn_mask
        ):
        """
        compute attention with given query, key, value.

        if attn_mask is given, apply attention mask.
        if is_causal is True, apply causal mask (attn_mask must be None).
        if use_relative_positional_symbols is True, the symbols are treated as relative positional embeddings.
            assumed to be of shape [len, len, dim] where len is the length of the sequence x.

        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape [bsz, len, d_model]
        symbols : torch.Tensor
            input tensor of shape [bsz, len, d_model] or [len, len, d_model] if use_relative_positional_symbols is True
        attn_mask : torch.Tensor, optional
            boolean attention mask of shape [len, len]. True at [i,j] indicates i is allowed to attend to j.
            By default None
        is_causal : bool, optional
            whether to apply a causal mask. If True, attn_mask must be None. By default False

        Returns
        -------
        torch.Tensor
            result of attention
        """

        bsz, seqlen, _ = x.shape

        # apply query/key projections (for attention) and reshape to split into different heads
        xq_attn, xk_attn = self.wq_attn(x), self.wk_attn(x)
        xq_attn = xq_attn.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk_attn = xk_attn.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # apply query/key projections (for relation) and reshape to split into different heads
        xq_rel, xk_rel = self.wq_rel(x), self.wk_rel(x)
        xq_rel = xq_rel.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk_rel = xk_rel.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # apply value projection to symbols
        sv = self.wv(symbols)

        if self.use_relative_positional_symbols:
            # make sure symbols are of shape [len, len, dim]
            assert symbols.shape[0] == symbols.shape[1] == seqlen
            sv = sv.view(seqlen, seqlen, self.n_kv_heads, self.head_dim)
        else:
            sv = sv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # grouped multiquery attention: expand out keys and values
        if self.n_rep_kv != 1:
            xk_attn = repeat_kv(xk_attn, self.n_rep_kv)  # (bs, seqlen, n_heads, head_dim)
            sv = repeat_kv(sv, self.n_rep_kv)  # (bs, seqlen, n_heads, head_dim)

        # make heads into a batch dimension
        xq_attn = xq_attn.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk_attn = xk_attn.transpose(1, 2)
        xq_rel = xq_rel.transpose(1, 2)
        xk_rel = xk_rel.transpose(1, 2)
        # sv: (seq_len, seq_len, n_heads, head_dim) or (bs, seq_len, n_heads, head_dim)

        assert not (attn_mask is not None and is_causal)
        if is_causal and attn_mask is None:
            attn_mask = compute_causal_mask(seqlen, device=xq_attn.device)
            # better to pass attn_mask rather than compute so that it doesn't need to be computed at each layer?

        # compute dot product for attn scores
        attn_scores = torch.matmul(xq_attn, xk_attn.transpose(2, 3)) * self.attn_scale # (bs, n_heads, seqlen, seqlen)

        # if softmax activation, masking is handled by adding -inf before softmax
        if attn_mask is not None:
            attn_mask_ = torch.zeros(seqlen, seqlen, dtype=xq_attn.dtype).masked_fill(attn_mask.logical_not(), float('-inf'))
            attn_scores = attn_scores + attn_mask_

        # apply (relation) activation to inner products
        attn_scores = nn.functional.softmax(attn_scores, dim=-1)

        # compute relations
        rel_scores = torch.matmul(xq_rel, xk_rel.transpose(2, 3)) * self.attn_scale
        rel_scores = self.rel_activation_(rel_scores)

        # for non-softmax activation, masking is handled by zero-ing out *after* activation
        if attn_mask is not None and self.rel_activation != 'softmax':
            attn_scores = attn_scores * attn_mask

        rca_scores = attn_scores * rel_scores
        rca_scores = self.attn_dropout(rca_scores)

        if not self.use_relative_positional_symbols:
            sv = sv.transpose(1, 2)
            output = torch.matmul(rca_scores, sv)  # (bs, n_heads, seqlen, head_dim)
        else:
            output = torch.einsum('bhij,ijhd->bihd', rca_scores, sv)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output, attn_scores, rel_scores