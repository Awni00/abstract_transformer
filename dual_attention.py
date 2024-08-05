import torch
import torch.nn as nn

from attention import Attention
from relational_attention import RelationalCrossAttention, DisentangledRelationalCrossAttention, RelationalAttention

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
        share_attn_params: bool = False,
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
        share_attn_params : bool, optional
            whether to share attention parameters between self-attention and relational attention.
            If True, w{q,k} in sensory attention and w{q,k}_attn in relational attention are shared.
            number of heads in each must be the same. By default False
        ra_type : str, optional
            type of relational attention module (e.g., whether to use RCA for an ablation experiment).
            by default 'relational_attention'.

        """
        super(DualAttention, self).__init__()
        self.d_model = d_model
        self.n_heads_sa = n_heads_sa
        self.n_heads_ra = n_heads_ra
        self.dropout = dropout
        self.sa_kwargs = sa_kwargs if sa_kwargs is not None else {}
        self.ra_kwargs = ra_kwargs if ra_kwargs is not None else {}
        self.ra_type = ra_type
        self.share_attn_params = share_attn_params

        if self.share_attn_params and n_heads_sa != n_heads_ra:
            raise ValueError("Number of heads in self-attention and relational attention must be the same if sharing attention parameters")

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
        elif self.use_rel_attn and ra_type=='rca':
            self.relational_attention = RelationalCrossAttention(
                d_model=d_model, n_heads=n_heads_ra,
                total_n_heads=self.total_n_heads, dropout=dropout,
                **self.ra_kwargs)
        elif self.use_rel_attn and ra_type=='disrca':
            self.relational_attention = DisentangledRelationalCrossAttention(
                d_model=d_model, n_heads=n_heads_ra,
                total_n_heads=self.total_n_heads, dropout=dropout,
                **self.ra_kwargs)
        elif self.use_rel_attn:
            raise ValueError(f"Invalid relational attention type: {ra_type}")

        if self.share_attn_params:
            self.self_attention.wq = self.relational_attention.wq_attn
            self.self_attention.wk = self.relational_attention.wk_attn


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
