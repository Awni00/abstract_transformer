"""
This file contains a self-contained single-file implementation of DualAttention and the DualAttention Transformer
as proposed in the paper:
"Disentangling and Integrating Relational and Sensory Information in Transformer Architectures"
Awni Altabaa, John Lafferty (2024). https://arxiv.org/abs/2405.16727

Author: Awni Altabaa
License: MIT License
"""

import torch
import torch.nn as nn

import math
from einops import rearrange

# An implementation of Dual Attention as proposed in the paper
# "Disentangling and Integrating Relational and Sensory Information in Transformer Architectures"
# Awni Altabaa, John Lafferty (2024). https://arxiv.org/abs/2405.16727

# The DualAttention module is a form of multi-head attention involving a composition of two distinct types of attention heads.
# The first type is standard self-attention, which captures object-level (i.e., sensory) features, and
# the second type is relational attention, which captures relational features.

# DualAttention is a concatenation of self-attention and relational attention heads.
class DualAttention(nn.Module):
    def __init__(self,
        d_model: int,
        n_heads_sa: int,
        n_heads_ra: int,
        dropout: float,
        sa_kwargs: dict = None,
        ra_kwargs: dict = None,
        ra_type: str = 'relational_attention'
    ):
        """An implementation of Dual Attention.

        The DualAttention module is a form of multi-head attention involving a composition of two distinct types of attention heads.
        The first type is standard self-attention, which captures object-level (i.e., sensory) features, and
        the second type is relational attention, which captures relational features.


        Parameters
        ----------
        d_model : int
            model dimension
        n_heads_sa : int
            number of self-attention heads
        n_heads_ra : int
            number of relational attention heads
        dropout : float
            dropout rate
        sa_kwargs : dict, optional
            self-attention kwargs, by default None
        ra_kwargs : dict, optional
            relational attention kwargs, by default None
        ra_type : str, optional
            type of relational attention module (e.g., whether to use RCA for an ablation experiment).
            by default 'relational_attention'.

        Raises
        ------
        ValueError
            _description_
        """
        super(DualAttention, self).__init__()
        self.d_model = d_model
        self.n_heads_sa = n_heads_sa
        self.n_heads_ra = n_heads_ra
        self.dropout = dropout
        self.sa_kwargs = sa_kwargs if sa_kwargs is not None else {}
        self.ra_kwargs = ra_kwargs if ra_kwargs is not None else {}
        self.ra_type = ra_type

        self.use_self_attn = n_heads_sa > 0
        self.use_rel_attn = n_heads_ra > 0


        self.total_n_heads = n_heads_sa + n_heads_ra

        if not (self.use_self_attn or self.use_rel_attn):
            raise ValueError("At least one of self-attention or relational attention must be used")

        if self.use_self_attn:
            self.self_attention = Attention(
                d_model=d_model, n_heads=n_heads_sa,
                total_n_heads=self.total_n_heads, dropout=dropout,
                **self.sa_kwargs)

        if self.use_rel_attn and ra_type=='relational_attention':
            self.relational_attention = RelationalAttention(
                d_model=d_model, n_heads=n_heads_ra,
                total_n_heads=self.total_n_heads, dropout=dropout,
                **self.ra_kwargs)
        # elif self.use_rel_attn and ra_type=='rca':
        #     self.relational_attention = RelationalCrossAttention(
        #         d_model=d_model, n_heads=n_heads_ra,
        #         total_n_heads=self.total_n_heads, dropout=dropout,
        #         **self.ra_kwargs)
        # elif self.use_rel_attn and ra_type=='disrca':
        #     self.relational_attention = DisentangledRelationalCrossAttention(
        #         d_model=d_model, n_heads=n_heads_ra,
        #         total_n_heads=self.total_n_heads, dropout=dropout,
        #         **self.ra_kwargs)
        else:
            raise ValueError(f"Invalid relational attention type: {ra_type}")


    def forward(
        self,
        x: torch.Tensor,
        symbols: torch.Tensor,
        attn_mask: torch.Tensor = None, # boolean attention mask: True indicates corresponding position *should* be attended to
        is_causal: bool = False, # indicates causal mask; should only set one of is_causal and attn_mask
        freqs_cos: torch.Tensor = None,
        freqs_sin: torch.Tensor = None,
        need_weights: bool = False # applies only to self-attention; determines whether FlashAttention is used or not
        ):

        # self-attention
        if self.use_self_attn:
            self_attn_out, self_attn_scores = self.self_attention(
                query=x, key=x, value=x,
                freqs_cos=freqs_cos, freqs_sin=freqs_sin,
                attn_mask=attn_mask, is_causal=is_causal,
                need_weights=need_weights)

        # relational cross-attention
        if self.use_rel_attn:
            rel_attn_out, *rel_attn_scores = self.relational_attention(
                x, symbols,
                attn_mask=attn_mask, is_causal=is_causal,
                freqs_cos=freqs_cos, freqs_sin=freqs_sin)

        # combine self-attention and relational cross-attention
        if self.use_rel_attn and self.use_self_attn:
            # concat self-attention output (E) and relational cross-attention output (A)
            out = torch.concat((self_attn_out, rel_attn_out), dim=-1)
        elif self.use_rel_attn:
            out = rel_attn_out # only use relational cross-attention
            self_attn_scores = None
        elif self.use_self_attn:
            out = self_attn_out # only use standard self-attention
            rel_attn_scores = None

        return out, self_attn_scores, rel_attn_scores



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
        """

        super().__init__()
        self.d_model = d_model # model dimension
        self.n_heads = n_heads # number of heads (for query)
        self.n_relations = n_relations if n_relations is not None else n_heads # number of relations
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads # n_kv_heads = 1 corresponds to multi-query attn
        self.rel_activation = rel_activation # "relation activation function"
        self.rel_activation_ = get_activation_function(rel_activation)
        self.symmetric_rels = symmetric_rels # whether to use symmetric relations
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv # whether to add bias to key/value projections
        self.add_bias_out = add_bias_out # whether to add bias to output projection
        self.use_relative_positional_symbols = use_relative_positional_symbols # whether to use relative positional symbols

        self.total_n_heads = n_heads if total_n_heads is None else total_n_heads # total number of heads in abstract attention
        self.key_dim = key_dim if key_dim is not None else self.d_model // self.total_n_heads # key dimension
        self.rel_proj_dim = rel_proj_dim if rel_proj_dim is not None else self.key_dim # dimension of relation projections

        self.n_rep_kv = self.n_heads // self.n_kv_heads # use same kv heads for several query heads
        self.head_dim = self.d_model // self.total_n_heads # dim of projections

        # make relative size of parameters and dimensions makes sense
        assert self.n_heads % self.n_kv_heads == 0, f"n_heads={self.n_heads}, n_kv_heads = {self.n_kv_heads}"
        assert self.n_rep_kv * self.n_kv_heads == self.n_heads, f"n_rep_kv={self.n_rep_kv}, n_kv_heads={self.n_kv_heads}, n_heads={self.n_heads}"
        assert self.total_n_heads * self.head_dim == self.d_model, f"total_n_heads={self.total_n_heads}, head_dim={self.head_dim}, d_model={self.d_model}"

        self.attn_scale = 1 / math.sqrt(self.head_dim) # for scaled dot product attention
        self.rel_scale = 1 / math.sqrt(self.rel_proj_dim)

        # Wq, Wk projections for attention
        self.wq_attn = nn.Linear(self.d_model, self.n_heads * self.key_dim, bias=False)
        self.wk_attn = nn.Linear(self.d_model, self.n_kv_heads * self.key_dim, bias=self.add_bias_kv)

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

# region Symbol Assignment Mechanisms

class SymbolicAttention(nn.Module):
    def __init__(self,
            d_model: int,
            n_heads: int,
            n_symbols: int,
            dropout: float = 0.0,
            scale: float = None,
            trainable_symbols: bool = True):
        """
        Symbolic Attention.

        Learns a library of "symbols" and corresponding template features.
        For a given input, retrieves a symbol from the symbol library via attention.

        Parameters
        ----------
        d_model : int
            model dimension. this is the dimension of the input and the dimension of the symbols and template features.
        n_heads : int
            number of heads in symbolic attention.
        n_symbols : int
            number of symbols in the symbol library.
        dropout : float, optional
            dropout probability, by default 0.0
        scale : float, optional
            scaling factor in scaled_dot_product_attention, by default None
        trainable_symbols: bool, optional
            whether to make the symbol library trainable, by default True
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_symbols = n_symbols
        self.dropout = dropout
        self.scale = scale
        self.trainable_symbols = trainable_symbols

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.template_features = nn.Parameter(torch.empty(self.n_symbols, self.d_model))
        self.symbol_library = nn.Parameter(torch.empty(self.n_symbols, self.d_model), requires_grad=trainable_symbols)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.template_features)
        torch.nn.init.normal_(self.symbol_library)


    def forward(self, x):

        batch_size, seq_len, dim = x.size()

        # create query from input
        query = self.q_proj(x)
        query = query.view(batch_size, seq_len, self.n_heads, dim // self.n_heads).transpose(1, 2)

        # create keys from template features
        key = self.template_features.view(self.n_symbols, self.n_heads, self.d_model // self.n_heads).transpose(0, 1)
        key = self._repeat_kv(key, batch_size)

        # create values from symbol library
        value = self.symbol_library.view(self.n_symbols, self.n_heads, self.d_model // self.n_heads).transpose(0, 1)
        value = self._repeat_kv(value, batch_size)

        retrieved_symbols = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            scale=self.scale, dropout_p=self.dropout, attn_mask=None, is_causal=False)
        retrieved_symbols = retrieved_symbols.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)

        return retrieved_symbols

    def _repeat_kv(self, x, batch_size):
        """
        template_features and symbol_library are of shape (n_heads, n_s, d_s//n_heads).
        repeat for each input and add a batch dimension of size batch_size.
        """
        return x.unsqueeze(0).repeat(batch_size, 1, 1, 1)

class PositionalSymbolRetriever(nn.Module):
    def __init__(self, symbol_dim, max_length, sinusoidal=False):
        """
        Postional Symbol Retriever.

        Learns a library of "symbols".
        Retrieves a symbol for each object based on its position.

        Parameters
        ----------
        symbol_dim : int
            dimension of the symbols.
        max_symbols : int
            maximum number of symbols.
        """

        super().__init__()
        self.symbol_dim = symbol_dim
        self.max_length = max_length
        self.sinusoidal = sinusoidal

        self.symbol_library = nn.Embedding(self.max_length, self.symbol_dim)

        # TODO: implement sinusoidal symbols?

    def forward(self, x):
        device = x.device
        batch_size, seq_len, dim = x.size()

        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        retrieved_symbols = self.symbol_library(pos).unsqueeze(0).repeat(batch_size, 1, 1)

        return retrieved_symbols

# TODO: add support for causal-only position-relative symbols?
# cuts param count by half
class PositionRelativeSymbolRetriever(nn.Module):
    def __init__(self, symbol_dim, max_rel_pos):
        """
        Position-Relative Symbol Retriever.

        For i -> j, the symbol s_{ij} encodes the relative position j - i.

        Parameters
        ----------
        symbol_dim : int
            dimension of the symbols.
        max_rel_pos : int
            maximum relative position encoded by symbols.
            Positions exceeding this will be truncated.
        """
        super().__init__()
        self.symbol_dim = symbol_dim
        self.max_rel_pos = max_rel_pos

        self.rel_pos_enc = RelativePositionalEncoding(dim=symbol_dim, max_rel_pos=max_rel_pos)

    def forward(self, x):
        length = x.shape[1]
        return self.rel_pos_enc(length)

class RelativePositionalEncoding(nn.Module):

    def __init__(self, dim: int, max_rel_pos: int):
        """
        module which returns relative positional embeddings for a given pair of sequences.

        I.e., returns tensor whose [i,j]-th entry is the embedding of the relative position "j-i"

        Parameters
        ----------
        dim : int
            dimension of embeddings
        max_rel_pos : int
            maximum relative position in either direction (used for clipping)
        """

        super().__init__()
        self.num_units = dim
        self.max_relative_position = max_rel_pos
        self.rel_pos_embeddings = nn.Parameter(torch.Tensor(max_rel_pos * 2 + 1, dim))
        nn.init.xavier_uniform_(self.rel_pos_embeddings)

    def forward(self, length_q, length_k=None):
        """
        Parameters
        ----------
        length_q : int
            length of query sequence
        length_k : _type_, optional
            length of key sequence, by default None

        Returns
        -------
        Tensor
            tensor of shape [length_q, length_k, dim] where [i,j] is the embedding of the relative position "j-i"
        """

        if length_k is None:
            length_k = length_q

        range_q = torch.arange(length_q) # TODO: need to set dtype or device?
        range_k = torch.arange(length_k)

        distance_mat = range_k[None, :] - range_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)

        final_mat = distance_mat_clipped + self.max_relative_position

        # final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.rel_pos_embeddings[final_mat] #.cuda()

        return embeddings

# endregion

# region Dual-Attention Blocks

class DualAttnEncoderBlock(nn.Module):

    def __init__(self,
            d_model: int,
            n_heads_sa: int,
            n_heads_ra: int,
            dff: int,
            activation: str,
            dropout_rate: float,
            norm_first: bool,
            norm_type: str = 'layernorm',
            sa_kwargs: dict = None,
            ra_kwargs: dict = None,
            ra_type: str = 'relational_attention',
            bias: bool = True,
            causal: bool = False):
        """
        Dual Attention Encoder Block.

        A Dual Attention Encoder is a variant of the Transformer Encoder that uses a combination of two distinct types of attention heads.
        The first type is standard self-attention, which captures object-level (i.e., sensory) features, and
        the second type is relational attention, which captures relational features.

        Parameters
        ----------
        d_model : int
            model dimension.
        n_heads_sa : int
            number of standard self-attention heads.
        n_heads_ra : int
            number of relational attention heads.
        dff : int
            intermediate dimension of feed-forward block.
        activation : str
            name of activation function to use in feedforward block.
        dropout_rate : float
            dropout rate.
        norm_first : bool
            whether to apply normalization before or after attention. norm_first=True means pre-norm otherwise post-norm.
        norm_type : 'layernorm' or 'rmsnorm, optional
            type of normalization to use, by default 'layernorm'
        sa_kwargs : dict, optional
            self-attention kwargs, by default None
        ra_kwargs : dict, optional
            relational attention kwargs, by default None
        ra_type : str, optional
            type of relational attention module (e.g., whether to use RCA for an ablation experiment), by default 'relational_attention'
        bias : bool, optional
            whether to use bias in multi-head attention, by default True
        causal : bool, optional
            whether attention operations should be causal, by default False
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads_sa = n_heads_sa
        self.n_heads_ra = n_heads_ra
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_first = norm_first
        self.norm_type = norm_type
        self.ra_type = ra_type
        self.bias = bias
        self.causal = causal

        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm1 = create_norm(self.d_model, self.norm_type)
        self.dual_attn = DualAttention(
            d_model=d_model, n_heads_sa=n_heads_sa, n_heads_ra=n_heads_ra,
            dropout=dropout_rate, sa_kwargs=sa_kwargs, ra_kwargs=ra_kwargs,
            ra_type=ra_type)

        self.norm2 = create_norm(self.d_model, self.norm_type)
        self.ff_block = FeedForwardBlock(self.d_model, dff=self.dff, activation=self.activation, use_bias=self.bias)

    # TODO: make attn_mask input so it only needs to be computed once?
    def forward(self, x, symbols, freqs_cos=None, freqs_sin=None):
        if self.norm_first:
            x = x + self._compute_dual_attn(self.norm1(x), symbols, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
            x = x + self._apply_ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._compute_dual_attn(x, symbols, freqs_cos=freqs_cos, freqs_sin=freqs_sin))
            x = self.dropout(x)
            x = self.norm2(x + self._apply_ff_block(x))
        return x

    def _compute_dual_attn(self, x, symbols, freqs_cos=None, freqs_sin=None):

        x, *_ = self.dual_attn(x, symbols,
            need_weights=False, is_causal=self.causal,
            freqs_cos=freqs_cos, freqs_sin=freqs_sin)

        x = self.dropout(x) # dropout

        return x

    def _apply_ff_block(self, x):
        x = self.ff_block(x)
        x = self.dropout(x)
        return x


class DualAttnDecoderBlock(nn.Module):
    def __init__(self,
                d_model: int,
                n_heads_sa: int,
                n_heads_ra: int,
                n_heads_cross: int,
                dff: int,
                activation: str,
                dropout_rate: float,
                norm_first: bool,
                norm_type: str = 'layernorm',
                sa_kwargs: dict = None,
                ra_kwargs: dict = None,
                cross_kwargs: dict = None,
                ra_type: str = 'relational_attention',
                bias: bool = True,
                causal: bool = True):
        """
        Dual Attention Decoder Block.

        A Dual Attention Decoder is a variant of the Transformer Decoder that uses a combination of two distinct types of attention heads.
        The first type is standard self-attention, which captures object-level (i.e., sensory) features, and
        the second type is relational attention, which captures relational features.

        Parameters
        ----------
        d_model : int
            model dimension.
        n_heads_sa : int
            number of standard self-attention heads.
        n_heads_ra : int
            number of relational attention heads.
        n_heads_cross : int
            number of cross-attention heads.
        dff : int
            intermediate dimension of feed-forward block.
        activation : str
            name of activation function to use in feedforward block.
        dropout_rate : float
            dropout rate.
        norm_first : bool
            whether to apply normalization before or after attention. norm_first=True means pre-norm otherwise post-norm.
        norm_type : 'layernorm' or 'rmsnorm, optional
            type of normalization to use, by default 'layernorm'
        sa_kwargs : dict, optional
            self-attention kwargs, by default None
        ra_kwargs : dict, optional
            relational attention kwargs, by default None
        cross_kwargs : dict, optional
            cross-attention kwargs, by default None
        ra_type : str, optional
            type of relational attention module (e.g., whether to use RCA for an ablation experiment), by default 'relational_attention'
        bias : bool, optional
            whether to use bias in multi-head attention, by default True
        causal : bool, optional
            whether attention operations should be causal, by default False
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads_sa = n_heads_sa
        self.n_heads_ra = n_heads_ra
        self.n_heads_cross = n_heads_cross
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_first = norm_first
        self.norm_type = norm_type
        self.ra_type = ra_type
        self.bias = bias
        self.causal = causal

        self.use_self_attn = n_heads_sa > 0
        self.use_rel_attn = n_heads_ra > 0

        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm1 = create_norm(self.d_model, self.norm_type)

        self.dual_attn = DualAttention(
            d_model=d_model, n_heads_sa=n_heads_sa, n_heads_ra=n_heads_ra,
            dropout=dropout_rate, sa_kwargs=sa_kwargs, ra_kwargs=ra_kwargs,
            ra_type=ra_type)

        self.norm2 = create_norm(self.d_model, self.norm_type)
        cross_kwargs = cross_kwargs if cross_kwargs is not None else {}
        self.cross_attn = Attention(
            self.d_model, self.n_heads_cross, dropout=self.dropout_rate,
            **cross_kwargs)
        self.norm3 = create_norm(self.d_model, self.norm_type)
        self.ff_block = FeedForwardBlock(self.d_model, dff=self.dff, activation=self.activation, use_bias=self.bias)

    def forward(self, x, context, symbols):
        if self.norm_first:
            x = x + self._compute_dual_attn(self.norm1(x), symbols)
            x = x + self._compute_cross_attn(self.norm2(x), context)
            x = x + self.ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._compute_dual_attn(x, symbols))
            x = self.norm2(x + self._compute_cross_attn(x, context))
            x = self.norm3(x + self.ff_block(x))
        return x

    def _compute_dual_attn(self, x, symbols):

        x, *_ = self.dual_attn(x, symbols, need_weights=False, is_causal=self.causal)

        x = self.dropout(x) # dropout

        return x

    def _compute_cross_attn(self, x, context):
        x = self.cross_attn(query=x, key=context, value=context, need_weights=False, is_causal=False)[0]
        x = self.dropout(x)
        return x

    def _apply_ff_block(self, x):
        x = self.ff_block(x)
        x = self.dropout(x)
        return x

# endregion

# region Dual-Attention language model

class DualAttnTransformerLM(nn.Module):
    """Dual Attention Transformer Language Model"""
    def __init__(self,
            vocab_size: int,
            d_model: int,
            n_layers: int,
            n_heads_sa: int,
            n_heads_ra: int,
            symbol_retrieval_kwargs: dict,
            dff: int,
            dropout_rate: float,
            activation: str,
            norm_first: bool,
            max_block_size: int,
            norm_type: str = 'layernorm',
            sa_kwargs: dict = None,
            ra_kwargs: dict = None,
            ra_type: str = 'relational_attention',
            symbol_retrieval: str = 'symbolic_attention',
            pos_enc_type: str = 'pos_emb',
            bias: bool = True):
        """
        Dual Attention Transformer Language Model.

        Parameters
        ----------
        vocab_size : int
            vocabulary size.
        d_model : int
            model dimension.
        n_layers : int
            number of layers.
        n_heads_sa : int
            number of self-attention heads in dual-attention.
        n_heads_ra : int
            number of relational attention heads in dual-attention.
        symbol_retrieval_kwargs : dict
            keyword arguments for symbol retrieval module.
        dff : int
            size of intermediate layer in feedforward blocks.
        dropout_rate : float
            dropout rate.
        activation : str
            name of activation function (e.g., 'relu', 'gelu', or 'swiglu').
        norm_first : bool
            whether to apply layer normalization before or after attention.
        max_block_size : int
            maximum context size.
        sa_kwargs : dict, optional
            keyword arguments for self-attention, by default None
        ra_kwargs : dict, optional
            keyword arguments for relational attention, by default None
        ra_type : 'relational_attention', 'rca', or 'disrca', optional
            type of relational attention module (e.g., whether to use RCA for an ablation experiment), by default 'relational_attention'
        symbol_retrieval : 'symbolic_attention', 'position_relative', 'positional_symbols', optional
            type of symbol retrieval module to use. this is shared across layers, by default 'symbolic_attention'
        pos_enc_type : 'pos_emb' or 'RoPE', optional
            type of positional encoding to use, by default 'pos_emb'
        bias : bool, optional
            whether to use bias in attention, by default True
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads_sa = n_heads_sa
        self.n_heads_ra = n_heads_ra
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_first = norm_first
        self.norm_type = norm_type
        self.block_size = max_block_size
        self.ra_type = ra_type
        self.symbol_retriever = symbol_retrieval
        self.pos_enc_type = pos_enc_type
        self.bias = bias

        self.n_heads = n_heads_sa + n_heads_ra

        if symbol_retrieval == 'symbolic_attention':
            symbol_retriever = SymbolicAttention(**symbol_retrieval_kwargs)
        # elif symbol_retrieval == 'rel_sym_attn':
            # symbol_retriever = RelationalSymbolicAttention(**symbol_retrieval_kwargs)
        elif symbol_retrieval == 'positional_symbols':
            symbol_retriever = PositionalSymbolRetriever(**symbol_retrieval_kwargs)
        elif symbol_retrieval == 'position_relative':
            symbol_retriever = PositionRelativeSymbolRetriever(**symbol_retrieval_kwargs)
        else:
            raise ValueError(
                f"`symbol_retrieval` must be one of 'symbolic_attention', 'rel_sym_attn', 'positional_symbols' or 'pos_relative."
                f"received {symbol_retrieval}")


        layers = dict(
            token_embedder = nn.Embedding(vocab_size, d_model),
            dropout = nn.Dropout(dropout_rate),
            symbol_retriever = symbol_retriever,
            blocks = nn.ModuleList([DualAttnEncoderBlock(
                d_model=d_model, n_heads_sa=n_heads_sa, n_heads_ra=n_heads_ra, dff=dff, dropout_rate=dropout_rate,
                activation=activation, norm_first=norm_first, norm_type=norm_type,
                sa_kwargs=sa_kwargs, ra_kwargs=ra_kwargs, ra_type=ra_type, causal=True)
                for _ in range(n_layers)]),
            norm = create_norm(d_model, norm_type),
            final_out = nn.Linear(d_model, vocab_size, bias=False)
            )

        if pos_enc_type == 'pos_emb':
            # if using positional embedding, create positional embedding layer
            positional_embedder = nn.Embedding(max_block_size, d_model)
            layers['positional_embedder'] = positional_embedder
        elif pos_enc_type == 'RoPE':
            # if using RoPE, precompute RoPE sine-cosine rotation matrices
            freqs_cos, freqs_sin = precompute_freqs_cis(self.d_model // self.n_heads, self.block_size)
            self.register_buffer("freqs_cos", freqs_cos, persistent=False)
            self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        else:
            raise ValueError('`pos_enc_type` invalid')

        self.layers = nn.ModuleDict(layers)

        # weight-tying embedder and final layer
        self.layers.token_embedder.weight = self.layers.final_out.weight

        # initialize weights
        self.apply(self._init_weights)
        # NOTE: previously, I did not apply special initialization, but it turns out that it is important


        # per-GPT2 paper, scale intialization of output projection and last layer of mlp
        # apply special n_layer-scaled initialization to layers that add to the residual stream
        # (output projection of attention and last layer of mlp)
        # this ensures that, at initialization, adding to the residual stream does not cause things to blow up
        # note: while the _init_weights seemed to have a big effect, it is unclear what effect this is having
        mlp_special_init_layer = 'linear3' if activation == 'swiglu' else 'linear2'
        for pn, p in self.named_parameters():
            if pn.endswith(f'{mlp_special_init_layer}.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            # NOTE: wr in relational attention is Parameter not Linear. do we need to init it the same way? FIXME
        elif isinstance(module, RelationalAttention):
            torch.nn.init.normal_(module.wr, mean=0.0, std=0.02) # wr is a nn.Parameter now so needs to be initialized separately
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        device = x.device
        b, t = x.size()
        assert t <= self.block_size, f'Input sequence length {t} exceeds maximum block size {self.block_size}'

        x = self.layers.token_embedder(x)

        if self.pos_enc_type == 'pos_emb':
            # if using positional embeddings, add positional embeeddings
            positions = torch.arange(0, t, dtype=torch.long, device=device)
            positional_embedding = self.layers.positional_embedder(positions)
            x = x + positional_embedding
            freqs_cos, freqs_sin = None, None # not using RoPE
        elif self.pos_enc_type == 'RoPE':
            # otherwise, get the RoPE matrices
            freqs_cos = self.freqs_cos[:t]
            freqs_sin = self.freqs_sin[:t]

        for block in self.layers.blocks:
            symbols = self.layers.symbol_retriever(x)
            x = block(x, symbols, freqs_cos=freqs_cos, freqs_sin=freqs_sin)

        x = self.layers.norm(x)

        if targets is not None:
            # compute loss if given targets
            logits = self.layers.final_out(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=-1)
        else:
            logits = self.layers.final_out(x[:, [-1], :])
            loss = None

        return logits, loss

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # NOTE: Model Flops Utilization (MFU) is a measure of how much of the peak FLOPS of the GPU is being utilized.
        # PaLM paper has computed this for standard Transformers
        # haven't done this yet for DualAttention Transformer, so this is a placeholder

        # # first estimate the number of flops we do per iteration.
        # # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        # print('WARNING: estimate_mfu implementation not checked for DualAttnTransformerLM')
        # N = self.get_num_params()
        # nh = self.n_heads_sa + self.n_heads_ra
        # L, H, Q, T = self.n_layers, nh, self.d_model//self.n_heads_sa, self.block_size
        # flops_per_token = 6*N + 12*L*H*Q*T
        # flops_per_fwdbwd = flops_per_token * T
        # flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # # express our flops throughput as ratio of A100 bfloat16 peak flops
        # flops_achieved = flops_per_iter * (1.0/dt) # per second
        # flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        # mfu = flops_achieved / flops_promised
        mfu = -1
        return mfu

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_enc_type=='pos_emb':
            n_params -= self.layers.positional_embedder.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None):
        """
        Generate max_new_tokens new tokens, conditioning on the input idx.

        Parameters
        ----------
        idx : Tensor[int]
            tensor of shape (batch_size, seq_len) with input tokens.
        max_new_tokens : int
            number of new tokens to generate
        temperature : float, optional
            temperature parameter of softmax, by default 1.0
        top_k : int, optional
            top-k sampling parameter, by default None

        Returns
        -------
        Tensor[int]
            tensor of shape (batch_size, seq_len + max_new_tokens) with generated tokens.
        """
        for _ in range(max_new_tokens):
            # crop the sequence if it is longer thanblock_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond) # forward pass
            logits = logits[:, -1, :] / temperature # scale by temperature

            # optionally, crop logits to top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.nn.functional.softmax(logits, dim=-1) # convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sample from distribution
            idx = torch.cat((idx, idx_next), dim=1) # append to sequence

        return idx

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    use_fused = (device_type == 'cuda')
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
    print(f"using fused AdamW: {use_fused}")

    return optimizer

# endregion

# region standard Transformer modules & utility functions
class Attention(nn.Module):
    def __init__(self,
            d_model: int,
            n_heads: int,
            dropout: float,
            key_dim: int = None,
            n_kv_heads: int = None,
            add_bias_kv: bool = False,
            add_bias_out: bool = False,
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


class FeedForwardBlock(nn.Module):

    def __init__(self,
            embed_dim: int,
            dff: int = None,
            activation: str = 'relu',
            use_bias: bool = False):
        """
        Feed-forward block.

        A 2-layer neural network with activation function in between.

        Parameters
        ----------
        embed_dim : int
            embedding dimension of input.
        dff : int, optional
            size of intermediate layer. if None, 4 * embed_dim.
        activation : str, optional
            name of activation function, by default 'relu'
        use_bias : bool, optional
            whether to use bias in linear layers, by default False
        """

        super().__init__()
        self.embed_dim = embed_dim

        # set dff according to activation function if not given
        if dff is None and activation == 'swiglu':
            self.dff = int(2/3 * 4 * embed_dim)
        elif dff is None:
            self.dff = 4 * embed_dim
        else:
            self.dff = dff

        self.use_bias = use_bias
        self.activation = activation
        if self.activation != 'swiglu':
            self.activation_ = get_activation_function(activation)

        self.linear1 = nn.Linear(self.embed_dim, self.dff, bias=self.use_bias)
        self.linear2 = nn.Linear(self.dff, self.embed_dim, bias=self.use_bias)
        if self.activation == 'swiglu':
            self.linear3 = nn.Linear(self.embed_dim, self.dff, bias=self.use_bias)

    def forward(self, x):
        if self.activation == 'swiglu':
            return self.linear2(nn.functional.silu(self.linear1(x)) * self.linear3(x))
        else:
            x = self.linear1(x)
            x = self.activation_(x)
            x = self.linear2(x)
            return x

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def create_norm(d_model, norm_type):
    if norm_type=='layernorm':
        return nn.LayerNorm(d_model)
    elif norm_type=='rmsnorm':
        return RMSNorm(d_model)
    elif norm_type=='none':
        return  nn.Identity()
    else:
        raise ValueError(f'norm_type {norm_type} not valid')

def get_activation_function(name):
    """gets activation function by its name."""

    activation_dict = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'gelu': nn.GELU(approximate='tanh'),
        'silu': nn.SiLU(),
        'softmax': nn.Softmax(dim=-1),
        'identity': nn.Identity(),
        # add more if needed
    }
    if name in activation_dict:
        return activation_dict[name]
    else:
        raise ValueError(f'Activation function {name} not found in {activation_dict.keys()}')


# Utilities associated with computing attention
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
    ):

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# NOTE: be careful. pytorch API is inconsistent about whether True means attend or not attend. 
# this works with the Attention module implemented above, but will only be compatible with some but not all pytorch implementations
# e.g., works with nn.functional.scaled_dot_product_attention but not nn.MultiHeadAttention
def compute_diag_mask(size, device=None):
    """computes an attention mask with False on the diagonal and True elsewhere"""

    diag_mask = torch.eye(size, device=device).logical_not()
    # diag_mask = diag_mask.masked_fill(diag_mask == 1, float('-inf'))
    return diag_mask

def compute_causal_mask(size, device=None):
    """computes an attention mask with True at (i,j) if i <= j"""
    causal_mask = torch.tril(torch.ones(size, size, device=device)).bool()
    return causal_mask


# endregion