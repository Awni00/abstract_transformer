"""
A module implementating relational attention.

This module also includes an implementation of relational cross-attention from our previous paper for ablation experiments.
"""

import torch
from torch import nn
from model_utils import get_activation_function
import math
from einops import rearrange

from attention_utils import repeat_kv, apply_rotary_emb, compute_causal_mask

# Implementation of RelationalAttention as proposed in
# > "Disentangling and Integrating Relational and Sensory Information in Transformer Architectures"
# > Awni Altabaa, John Lafferty (2024). https://arxiv.org/abs/2405.16727

# Relational attention defines a differentiable information-retrieval operation where the information retrieved
# is the relations between objects. The "message" sent from one object to another is the relations between the
# sender and the receiver, tagged with a symbol identifying the sender. These messages are aggregated based on the
# receiver's features via softmax attention scores.

# Relational attention takes the form
# Math: \mathrm{RelAttn}(x_1, ..., x_n) = \sum_{j} \alpha_{ij} (r(x_i, x_j) W_r + s_j W_s)
# Math: \alpha = \mathrm{Softmax}((x W_q^{attn}) (x W_k^{attn})^\intercal)
# Math: r(x_i, x_j) = (\langle x_i W_{q, \ell}^{rel}, x_j W_{k, \ell}^{rel}\rangle)_{\ell \in [d_r]}
# Math: (s_1, ..., s_n) = \mathrm{SymbolRetriever}(x_1, ..., x_n)

class RelationalAttention(nn.Module):
    def __init__(self,
            d_model: int,
            n_heads: int,
            n_relations: int = None,
            dropout: float = 0.0,
            key_dim: int = None,
            n_kv_heads: int = None,
            rel_activation: str = 'identity',
            rel_proj_dim: int = None,
            add_bias_kv: bool = False,
            add_bias_out: bool = False,
            total_n_heads: int = None,
            symmetric_rels: bool = False,
            symmetric_attn: bool = False,
            use_relative_positional_symbols: bool = False
            ):
        """
        An implementation of Relational Attention (RA).

        Relational attention defines a differentiable information-retrieval operation where the information retrieved
        is the relations between objects. The "message" sent from one object to another is the relations between the
        sender and the receiver, tagged with a symbol identifying the sender. These messages are aggregated based on the
        receiver's features via softmax attention scores.

        The learnable parameters include a set of query/key projections which determine the attention scores, and hence
        the ``selection criteria'', as well as a set of query/key projections for computing relations between objects.
        They also include per-head projections for the symbols and relations, as well as a final output projection.

        This module supports symmetric relations, position-relative symbolic embeddings,
        multi-query attention/grouped query attention, and control over total number of heads (for use with "dual attention").

        Parameters
        ----------
        d_model : int
            model dimension
        n_heads : int
            number of attention heads (query heads if n_kv_heads is set)
        n_relations : int, optional
            number of relations. If None, n_relations = n_heads. By default None
        dropout : float, optional
            dropout rate. By default 0.0
        n_kv_heads : int, optional
            number of key/value heads. used to implement multi-query attention or grouped query attention.
            n_kv_heads=1 corresponds to MQA, n_kv_heads > 1 corresponsd to grouped query attention.
            n_kv_heads=n_heads is standard MHA. uses MHA when None. By default None
        rel_activation : str, optional
            name of activation function applied to relations. By default 'identity'.
        rel_proj_dim : int, optional
            dimension of relation projections. If None, rel_proj_dim = d_model // n_relations. By default None.
        add_bias_kv : bool, optional
            whether to use bias in key/value projections, by default False
        add_bias_out : bool, optional
            whether to use bias in out projection, by default False
        total_n_heads : int, optional
            total number of heads in dual attention (if using dual attention).
            used to ensure that concat(A, E) is of dimension d_model after concatentation.
            hence, output dimension is (d_model // total_heads) * n_heads.
            if None, total_heads = n_heads and output dimension is d_model
        symmetric_rels : bool, optional
            whether to weight-tie the query and key projections for relations, making symmetric relations.
            By default False
        symmetric_attn : bool, optional
            whether to weight-tie the query and key projections for attention, making symmetric attention.
            By default False
        use_relative_positional_symbols : bool, optional
            whether to use relative positional symbols, by default False
        """

        super().__init__()
        self.d_model = d_model # model dimension
        self.n_heads = n_heads # number of heads (for query)
        self.n_relations = n_relations if n_relations is not None else n_heads # number of relations
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads # n_kv_heads = 1 corresponds to multi-query attn
        self.rel_activation = rel_activation # "relation activation function"
        self.rel_activation_ = get_activation_function(rel_activation)
        self.symmetric_rels = symmetric_rels # whether to use symmetric relations
        self.symmetric_attn = symmetric_attn
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv # whether to add bias to key/value projections
        self.add_bias_out = add_bias_out # whether to add bias to output projection
        self.use_relative_positional_symbols = use_relative_positional_symbols # whether to use relative positional symbols

        self.total_n_heads = n_heads if total_n_heads is None else total_n_heads # total number of heads in abstract attention
        self.head_dim = self.d_model // self.total_n_heads # dim of projections
        self.n_rep_kv = self.n_heads // self.n_kv_heads # use same kv heads for several query heads
        self.key_dim = key_dim if key_dim is not None else self.head_dim # key dimension
        self.rel_proj_dim = rel_proj_dim if rel_proj_dim is not None else self.head_dim # dimension of relation projections

        # make relative size of parameters and dimensions makes sense
        assert self.n_heads % self.n_kv_heads == 0, f"n_heads={self.n_heads}, n_kv_heads = {self.n_kv_heads}"
        assert self.n_rep_kv * self.n_kv_heads == self.n_heads, f"n_rep_kv={self.n_rep_kv}, n_kv_heads={self.n_kv_heads}, n_heads={self.n_heads}"
        assert self.total_n_heads * self.head_dim == self.d_model, f"total_n_heads={self.total_n_heads}, head_dim={self.head_dim}, d_model={self.d_model}"
        assert self.rel_proj_dim * self.n_relations == self.head_dim * self.n_heads, f"rel_proj_dim={self.rel_proj_dim}, n_relations={self.n_relations}, head_dim={self.head_dim}"

        self.attn_scale = 1 / math.sqrt(self.head_dim) # for scaled dot product attention
        self.rel_scale = 1 / math.sqrt(self.rel_proj_dim) # for relations

        # Wq, Wk projections for attention
        self.wq_attn = nn.Linear(self.d_model, self.n_heads * self.key_dim, bias=False)
        self.wk_attn = nn.Linear(self.d_model, self.n_kv_heads * self.key_dim, bias=self.add_bias_kv)
        if self.symmetric_attn:
            self.wk_attn = self.wq_attn

        # Wq, Wk projections for relation
        self.wq_rel = nn.Linear(self.d_model, self.n_relations * self.rel_proj_dim, bias=False)
        if self.symmetric_rels:
            self.wk_rel = self.wq_rel
        else:
            self.wk_rel = nn.Linear(self.d_model, self.n_relations * self.rel_proj_dim, bias=False)
        # NOTE: attn Wq, Wk have n_kv_heads param for multi-query/grouped-query attention
        # but rel Wq, Wk do not have this param. TODO: think about whether we want to adjust implementation?

        # W_r = (W_r^h)_h projection mapping r(x_i, x_j) to common dimension with symbols
        self.wr = nn.Parameter(torch.empty(self.n_heads, self.head_dim, self.n_relations))
        torch.nn.init.kaiming_uniform_(self.wr, a=math.sqrt(5))
        # W_s = (W_s^h)_h = W_v projection for symbols
        self.wv = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=self.add_bias_kv)
        # Final output projection
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.n_heads * self.head_dim, bias=self.add_bias_out)
        self.attn_dropout = nn.Dropout(self.dropout) # dropout for attention scores
        self.resid_dropout = nn.Dropout(self.dropout) # dropout for output

    def forward(
        self,
        x: torch.Tensor,
        symbols: torch.Tensor,
        freqs_cos: torch.Tensor = None,
        freqs_sin: torch.Tensor = None,
        attn_mask: torch.Tensor = None, # boolean attention mask: True indicates corresponding position *should* be attended to
        is_causal: bool = False # indicates causal mask (will be computed automatically); should only set one of is_causal and attn_mask
        ):
        """
        compute attention with given query, key, value.

        if freqs_cos and freqs_sin are given, apply rotary positional embeddings.
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
        freqs_cos : torch.Tensor, optional
            cosine of frequencies for RoPE. RoPE is applied if given. By default None
        freqs_sin : torch.Tensor, optional
            cosine of frequencies for RoPE. RoPE is applied if given. By default None
        attn_mask : torch.Tensor, optional
            boolean attention mask of shape [len, len]. True at [i,j] indicates i is allowed to attend to j.
            By default None
        is_causal : bool, optional
            whether to apply a causal mask. If True, attn_mask must be None. By default False

        Returns
        -------
        tuple[torch.Tensor]
            outputs [bsz, len, d_model], attention scores [bsz, n_heads, len, len], relations [bsz, len, len, n_relations]
        """

        bsz, seqlen, _ = x.shape

        ## compute attention scores
        # apply query/key projections for attention and reshape to split into different heads
        xq_attn, xk_attn = self.wq_attn(x), self.wk_attn(x) # shape: [bsz, seqlen, d_model] (d_model = n_heads * head_dim)
        xq_attn = rearrange(xq_attn, 'b l (nh hd) -> b l nh hd', nh=self.n_heads) # shape: [bsz, seqlen, n_heads, head_dim]
        xk_attn = rearrange(xk_attn, 'b l (nh hd) -> b l nh hd', nh=self.n_kv_heads) # shape: [bsz, seqlen, n_kv_heads, head_dim]

        # apply RoPE relative positional embeddings (if given)
        if freqs_cos is not None and freqs_sin is not None:
            xq_attn, xk_attn = apply_rotary_emb(xq_attn, xk_attn, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        if self.n_rep_kv != 1:
            xk_attn = repeat_kv(xk_attn, self.n_rep_kv)  # (bs, seqlen, n_heads, head_dim)

        # transpose for matmul, make heads into a batch dimension
        xq_attn = xq_attn.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk_attn = xk_attn.transpose(1, 2) # (bs, n_heads, seqlen, head_dim)

        assert not (attn_mask is not None and is_causal) # attn_mask must not be given if is_causal
        # if is_causal create attn_mask
        if is_causal and attn_mask is None:
            attn_mask = compute_causal_mask(seqlen, device=xq_attn.device)

        # compute dot product for attn scores
        # Math: \alpha_{ij}^h = \langle W_q^{attn,h} x_i, W_k^{attn,h} x_j \rangle
        attn_scores = torch.matmul(xq_attn, xk_attn.transpose(2, 3)) * self.attn_scale # (bs, n_heads, seqlen, seqlen)

        # TODO: instead of creating a mask each time, it can be added to the buffer using a max_seq_len argument
        # e.g., see: https://github.com/karpathy/llama2.c/blob/master/model.py
        if attn_mask is not None:
            attn_mask_ = torch.zeros(seqlen, seqlen, dtype=xq_attn.dtype, device=xq_attn.device).masked_fill(attn_mask.logical_not(), float('-inf'))
            attn_scores = attn_scores + attn_mask_

        # apply (relation) activation to inner products
        attn_scores = nn.functional.softmax(attn_scores, dim=-1) # (bs, n_heads, seqlen, seqlen)
        attn_scores = self.attn_dropout(attn_scores)
        # NOTE: does it make sense to dropout attention scores?
        # it's done in Vaswani et al's original implementation and continues to be used, but standard dropout is not "closed under" simplex...

        ## compute relations
        # apply query/key projections for relation and reshape to split into different heads
        xq_rel, xk_rel = self.wq_rel(x), self.wk_rel(x) # shape: [bsz, seqlen, n_rels * rel_proj_dim]
        xq_rel = rearrange(xq_rel, 'b l (nr rd) -> b l nr rd', nr=self.n_relations) # shape: [bsz, seqlen, n_relations, rel_proj_dim]
        xk_rel = rearrange(xk_rel, 'b l (nr rd) -> b l nr rd', nr=self.n_relations) # shape: [bsz, seqlen, n_relations, rel_proj_dim]

        # apply value projection to symbols
        sv = self.wv(symbols) # shape: [bsz, seqlen, d_model] or [seqlen, seqlen, d_model] if use_relative_positional_symbols
        # grouped multiquery attention: expand out keys and values

        if self.use_relative_positional_symbols:
            # make sure symbols are of shape [len, len, dim]
            assert symbols.shape[0] == symbols.shape[1] == seqlen, f"symbols must be of shape [len, len, dim], received {symbols.shape}"
            sv = rearrange(sv, 'l1 l2 (nh hd) -> l1 l2 nh hd', nh=self.n_kv_heads) # shape: [seqlen, seqlen, n_kv_heads, head_dim]
        else:
            sv = rearrange(sv, 'b l (nh hd) -> b l nh hd', nh=self.n_kv_heads) # shape: [bsz, seqlen, n_kv_heads, head_dim]

        if self.n_rep_kv != 1:
            sv = repeat_kv(sv, self.n_rep_kv)  # (bs, seqlen, n_heads, head_dim)

        xq_rel = xq_rel.transpose(1, 2) # (bs, n_heads, seqlen, head_dim)
        xk_rel = xk_rel.transpose(1, 2) # (bs, n_heads, seqlen, head_dim)
        # sv: (seq_len, seq_len, n_heads, head_dim) or (bs, seq_len, n_heads, head_dim)

        # compute relations
        # Math: r(x_i, x_j) = (\langle W_q^{rel,\ell} x_i, W_k^{rel,\ell} x_j \rangle)_{\ell \in [d_r]}
        relations = torch.matmul(xq_rel, xk_rel.transpose(2, 3)) * self.rel_scale
        relations = self.rel_activation_(relations) # (bs, n_rels, seqlen, seqlen)

        # transpose to put "heads"/"relations" in final dim
        relations = rearrange(relations, 'b nr i j -> b i j nr') # (bs, seqlen, seqlen, n_rels)

        # NOTE: in a previous version of this implementation, the relations were mapped to head_dim-dimensional space with W_r^h
        # *before* the attention operation. However, this requires manifesting a large (N * N * D)- dimensional tensor instead of
        # an (N * N * R)-dimensional tensor (where R << D; R usually equals n_heads). This is a significant memory bottleneck.
        # This caused the memory requirement to scale quadratically with the sequence length which was prohibitive
        # Here, instead, we only manifest the (N * N * R)-dimensional tensor, compute attention over the relations to obtain an (N * H * R)-dimensional tensor,
        # then project to the final (N * H * head_dim)-dimensional tensor. This is much more memory efficient.

        # compute disentangled relational cross attention
        if not self.use_relative_positional_symbols:
            # sv: (bs, seqlen, n_heads, head_dim)
            # attn_scores: (bs, n_heads, seqlen, seqlen)
            # relations: (bs, seqlen, seqlen, n_heads, head_dim)
            # Math: A_i^h = \sum_j \alpha_{ij}^h (r(x_i, x_j) W_r^h + s_j W_s^h)

            # attend to symbols for each head
            attended_symbols = torch.einsum('bhij,bjhd->bihd', attn_scores, sv) # (bs, seqlen, n_heads, head_dim)

            # attend to relations for each head
            # Math: \sum_j \alpha_{ij}^h r(x_i, x_j)
            attended_relations = torch.einsum('bhij,bijr->bihr', attn_scores, relations) # (bs, seqlen, n_heads, n_relations)

            # project relations to common dimension with symbols (per-head)
            # Math: W_r^h (attended_relations)
            attended_relations = torch.einsum('bihr,hdr->bihd', attended_relations, self.wr) # (bs, seqlen, n_heads, head_dim)

            output = attended_symbols + attended_relations # (bs, seqlen, n_heads, head_dim)
        else:
            # sv: (seqlen, seqlen, n_heads, head_dim)
            # attn_scores: (bs, n_heads, seqlen, seqlen)
            # relations: (bs, seqlen, seqlen, n_heads, head_dim)
            # Math: A_i^h = \sum_j \alpha_{ij}^h (r(x_i, x_j) W_r^h + s_{j-i} W_s)

            # attend to symbols for each head
            attended_symbols = torch.einsum('bhij,ijhd->bihd', attn_scores, sv) # (bs, seqlen, n_heads, head_dim)

            # Math: \sum_j \alpha_{ij}^h r(x_i, x_j)
            attended_relations = torch.einsum('bhij,bijr->bihr', attn_scores, relations) # (bs, seqlen, n_heads, n_relations)

            # project relations to common dimension with symbols (per-head)
            # Math: W_r^h (attended_relations)
            attended_relations = torch.einsum('bihr,hdr->bihd', attended_relations, self.wr) # (bs, seqlen, n_heads, head_dim)

            output = attended_symbols + attended_relations # (bs, seqlen, n_heads, head_dim)

        # concat heads
        # Math: A_i = \mathrm{concat}(A_i^1, ..., A_i^{n_h})
        output = rearrange(output, 'b l nh hd -> b l (nh hd)') # (bs, seqlen, n_heads * head_dim)

        # final projection into the residual stream
        # Math: A_i \gets W_o A_i
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output, attn_scores, relations



# An implementation of Relational Cross Attention (RCA) from the paper
# > "Abstractors and relational cross-attention: An inductive bias for explicit relational reasoning in Transformers"
# > Awni Altabaa, Taylor Webb, Jonathan Cohen, John Lafferty. ICLR (2024). https://arxiv.org/abs/2304.00195
# This implementation adds a few addition options compared to the implementation in the original project repo: github.com/awni00/abstractor.

# Relational Cross-Attention Takes the form
# Math: \mathrm{RCA}(x_1, ..., x_n) = \sum_{j} \alpha_{ij} s_{j}
# Math: \alpha = \mathrm{Softmax}((x W_q) (x W_k)^\intercal)
# Math: (s_1, \ldots, s_n) = \mathrm{SymbolRetriever}(x_1, \ldots, x_n)

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

        This corresponds to RCA as proposed by
        "Abstractors and relational cross-attention: An inductive bias for explicit relational reasoning in Transformers"
        Awni Altabaa, Taylor Webb, Jonathan Cohen, John Lafferty. ICLR (2024). https://arxiv.org/abs/2304.00195

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
        freqs_cos: torch.Tensor = None,
        freqs_sin: torch.Tensor = None,
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
        freqs_cos : torch.Tensor, optional
            cosine of frequencies for RoPE. RoPE is applied if given. By default None
        freqs_sin : torch.Tensor, optional
            cosine of frequencies for RoPE. RoPE is applied if given. By default None
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
        xq, xk, sv = self.wq(x), self.wk(x), self.wv(symbols)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)


        if self.use_relative_positional_symbols:
            # make sure symbols are of shape [len, len, dim]
            assert symbols.shape[0] == symbols.shape[1] == seqlen
            sv = sv.view(seqlen, seqlen, self.n_kv_heads, self.head_dim)
        else:
            sv = sv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # apply RoPE relative positional embeddings (if given)
        if freqs_cos is not None and freqs_sin is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        if self.n_rep_kv != 1:
            xk = repeat_kv(xk, self.n_rep_kv)  # (bs, seqlen, n_heads, head_dim)
            sv = repeat_kv(sv, self.n_rep_kv)  # (bs, seqlen, n_heads, head_dim)

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
            attn_mask_ = torch.zeros(seqlen, seqlen, dtype=xq.dtype, device=xq.device).masked_fill(attn_mask.logical_not(), float('-inf'))
            scores = scores + attn_mask_

        # apply (relation) activation to inner products
        scores = self.activation_(scores)

        # for non-softmax activation, masking is handled by zero-ing out *after* activation
        if attn_mask is not None and self.activation != 'softmax':
            scores = scores * attn_mask

        scores = self.attn_dropout(scores)


        if not self.use_relative_positional_symbols:
            sv = sv.transpose(1, 2)
            output = torch.matmul(scores, sv)  # (bs, n_heads, seqlen, head_dim)
            output = output.transpose(1, 2) # (bs, seqlen, n_heads, head_dim)
        else:
            output = torch.einsum('bhij,ijhd->bihd', scores, sv)

        # concat heads
        output = output.contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output, scores

# Below is an implementation of an iteration over RCA which attempts to disentangle the attention and relation operations.
# It was later improved to the RelationalAttention module, which was what we focus on in the paper.
# Disentangled RCA uses two sets of learned projections, one for attention and one for computing relations.

# Disentangled Relational Cross-Attention
# Math: \mathrm{DisRCA}(x_1, ..., x_n) = \sum_{j} \alpha_{ij} r(x_i, x_j) s_j
# Math: \alpha = \mathrm{Softmax}((x W_q^{attn}) (x W_k^{attn})^\intercal)
# Math: r(x_i, x_j) = x_i W_{q}^{rel}) (x_j W_{k}^{rel})^\intercal
# Math: (s_1, ..., s_n) = \mathrm{SymbolRetriever}(x_1, ..., x_n)

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
        rel_activation : str, optional
            name of activation function applied to relations. By default 'identity'.
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
        freqs_cos: torch.Tensor = None,
        freqs_sin: torch.Tensor = None,
        attn_mask: torch.Tensor = None, # boolean attention mask: True indicates corresponding position *should* be attended to
        is_causal: bool = False # indicates causal mask; should only set one of is_causal and attn_mask
        ):
        """
        compute attention with given query, key, value.

        if freqs_cos and freqs_sin are given, apply rotary positional embeddings.
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
        freqs_cos : torch.Tensor, optional
            cosine of frequencies for RoPE. RoPE is applied if given. By default None
        freqs_sin : torch.Tensor, optional
            cosine of frequencies for RoPE. RoPE is applied if given. By default None
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

        # apply RoPE relative positional embeddings (if given)
        if freqs_cos is not None and freqs_sin is not None:
            xq_attn, xk_attn = apply_rotary_emb(xq_attn, xk_attn, freqs_cos, freqs_sin)

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

        assert not (attn_mask is not None and is_causal) # attn_mask must not be given if is_causal
        # if is_causal create attn_mask
        if is_causal and attn_mask is None:
            attn_mask = compute_causal_mask(seqlen, device=xq_attn.device)
            # better to pass attn_mask rather than compute so that it doesn't need to be computed at each layer?

        # compute dot product for attn scores
        attn_scores = torch.matmul(xq_attn, xk_attn.transpose(2, 3)) * self.attn_scale # (bs, n_heads, seqlen, seqlen)

        # TODO: instead of creating a mask each time, it can be added to the buffer using a max_seq_len argument
        # e.g., see: https://github.com/karpathy/llama2.c/blob/master/model.py
        if attn_mask is not None:
            attn_mask_ = torch.zeros(seqlen, seqlen, dtype=xq_attn.dtype, device=xq_attn.device).masked_fill(attn_mask.logical_not(), float('-inf'))
            attn_scores = attn_scores + attn_mask_

        # apply (relation) activation to inner products
        attn_scores = nn.functional.softmax(attn_scores, dim=-1)

        # compute relations
        rel_scores = torch.matmul(xq_rel, xk_rel.transpose(2, 3)) * self.attn_scale
        rel_scores = self.rel_activation_(rel_scores)

        rca_scores = attn_scores * rel_scores
        rca_scores = self.attn_dropout(rca_scores)

        if not self.use_relative_positional_symbols:
            sv = sv.transpose(1, 2)
            output = torch.matmul(rca_scores, sv)  # (bs, n_heads, seqlen, head_dim)
            output = output.transpose(1, 2) # (bs, seqlen, n_heads, head_dim)
        else:
            output = torch.einsum('bhij,ijhd->bihd', rca_scores, sv)

        # concat heads
        output = output.contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output, attn_scores, rel_scores
