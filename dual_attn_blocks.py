import torch
import torch.nn as nn
from dual_attention import DualAttention
from attention import Attention
from transformer_blocks import FeedForwardBlock, create_norm

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
            share_attn_params: bool = False,
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
        share_attn_params : bool, optional
            whether to share attention parameters between self-attention and relational attention.
            If True, w{q,k} in sensory attention and w{q,k}_attn in relational attention are shared.
            number of heads in each must be the same. By default False
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
        self.share_attn_params = share_attn_params
        self.bias = bias
        self.causal = causal

        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm1 = create_norm(self.d_model, self.norm_type)
        self.dual_attn = DualAttention(
            d_model=d_model, n_heads_sa=n_heads_sa, n_heads_ra=n_heads_ra,
            dropout=dropout_rate, sa_kwargs=sa_kwargs, ra_kwargs=ra_kwargs,
            ra_type=ra_type, share_attn_params=share_attn_params)

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
                share_attn_params: bool = False,
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
        share_attn_params : bool, optional
            whether to share attention parameters between self-attention and relational attention.
            If True, w{q,k} in sensory attention and w{q,k}_attn in relational attention are shared.
            number of heads in each must be the same. By default False
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
        self.share_attn_params = share_attn_params
        self.bias = bias
        self.causal = causal

        self.use_self_attn = n_heads_sa > 0
        self.use_rel_attn = n_heads_ra > 0

        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm1 = create_norm(self.d_model, self.norm_type)

        self.dual_attn = DualAttention(
            d_model=d_model, n_heads_sa=n_heads_sa, n_heads_ra=n_heads_ra,
            dropout=dropout_rate, sa_kwargs=sa_kwargs, ra_kwargs=ra_kwargs,
            ra_type=ra_type, share_attn_params=share_attn_params)

        self.norm2 = create_norm(self.d_model, self.norm_type)
        cross_kwargs = cross_kwargs if cross_kwargs is not None else {}
        self.cross_attn = Attention(
            self.d_model, self.n_heads_cross, dropout=self.dropout_rate,
            **cross_kwargs)
        self.norm3 = create_norm(self.d_model, self.norm_type)
        self.ff_block = FeedForwardBlock(self.d_model, dff=self.dff, activation=self.activation, use_bias=self.bias)

    def forward(self, x, context, symbols, freqs_cos=None, freqs_sin=None):
        if self.norm_first:
            x = x + self._compute_dual_attn(self.norm1(x), symbols, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
            x = x + self._compute_cross_attn(self.norm2(x), context)
            x = x + self.ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._compute_dual_attn(x, symbols, freqs_cos=freqs_cos, freqs_sin=freqs_sin))
            x = self.norm2(x + self._compute_cross_attn(x, context))
            x = self.norm3(x + self.ff_block(x))
        return x

    def _compute_dual_attn(self, x, symbols, freqs_cos=None, freqs_sin=None):
        x, *_ = self.dual_attn(x, symbols, need_weights=False, is_causal=self.causal, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
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
