import torch
import torch.nn as nn

from attention import Attention
from relational_cross_attention import RelationalCrossAttention, DisentangledRelationalCrossAttention

class AbstractAttention(nn.Module):
    def __init__(self,
        d_model: int,
        n_heads_sa: int,
        n_heads_rca: int,
        dropout: float,
        sa_kwargs: dict = None,
        rca_kwargs: dict = None,
        rca_disentangled: bool = False
    ):
        super(AbstractAttention, self).__init__()
        self.d_model = d_model
        self.n_heads_sa = n_heads_sa
        self.n_heads_rca = n_heads_rca
        self.dropout = dropout
        self.sa_kwargs = sa_kwargs if sa_kwargs is not None else {}
        self.rca_kwargs = rca_kwargs if rca_kwargs is not None else {}
        self.rca_disentangled = rca_disentangled

        self.use_self_attn = n_heads_sa > 0
        self.use_rca = n_heads_rca > 0


        self.total_n_heads = n_heads_sa + n_heads_rca

        if not (self.use_self_attn or self.use_rca):
            raise ValueError("At least one of self-attention or relational cross-attention must be used")

        if self.use_self_attn:
            self.self_attention = Attention(
                d_model=d_model, n_heads=n_heads_sa,
                total_n_heads=self.total_n_heads, dropout=dropout,
                **self.sa_kwargs)

        if self.use_rca and rca_disentangled:
            self.relational_cross_attention = DisentangledRelationalCrossAttention(
                d_model=d_model, n_heads=n_heads_rca,
                total_n_heads=self.total_n_heads, dropout=dropout,
                **self.rca_kwargs)
        elif self.use_rca and not rca_disentangled:
            self.relational_cross_attention = RelationalCrossAttention(
                d_model=d_model, n_heads=n_heads_rca,
                total_n_heads=self.total_n_heads, dropout=dropout,
                **self.rca_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        symbols: torch.Tensor,
        attn_mask: torch.Tensor = None, # boolean attention mask: True indicates corresponding position *should* be attended to
        is_causal: bool = False, # indicates causal mask; should only set one of is_causal and attn_mask
        freqs_cos: torch.Tensor = None,
        freqs_sin: torch.Tensor = None,
        need_weights: bool = False
        ):

        # self-attention
        if self.use_self_attn:
            self_attn_out, self_attn_scores = self.self_attention(
                query=x, key=x, value=x,
                freqs_cos=freqs_cos, freqs_sin=freqs_sin,
                attn_mask=attn_mask, is_causal=is_causal,
                need_weights=need_weights)

        # relational cross-attention
        if self.use_rca:
            rca_out, *rca_scores = self.relational_cross_attention(
                x, symbols,
                attn_mask=attn_mask, is_causal=is_causal)

        # combine self-attention and relational cross-attention
        if self.use_rca and self.use_self_attn:
            # concat self-attention output (E) and relational cross-attention output (A)
            out = torch.concat((self_attn_out, rca_out), dim=-1)
        elif self.use_rca:
            out = rca_out # only use relational cross-attention
            self_attn_scores = None
        elif self.use_self_attn:
            out = self_attn_out # only use standard self-attention
            rca_scores = None

        return out, self_attn_scores, rca_scores