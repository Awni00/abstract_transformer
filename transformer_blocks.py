import torch
import torch.nn as nn
from attention import Attention
import model_utils

class EncoderBlock(nn.Module):

    def __init__(self,
            d_model: int,
            n_heads: int,
            dff: int,
            dropout_rate: float,
            activation: str,
            norm_first: bool,
            norm_type: str = 'layernorm',
            bias: bool = True,
            causal: bool = False):
        """
        A Transformer Encoder Block.

        Consists of Self-attention, Feed-forward block and LayerNorms/Residuals.

        Parameters
        ----------
        d_model : int
            model dimension.
        n_heads : int
            number of self-attention heads.
        dff : int
            intermediate dimension of feed-forward block.
        dropout_rate : float
            dropout rate.
        activation : str
            name of activation function to use in feed-forward block.
        norm_first : bool
            whether to apply layer normalization before or after attention.
        norm_type: str, optional
            type of normalization to use. 'layernorm' or 'rmsnorm'. Default is 'layernorm'.
        bias : bool, optional
            whether to use bias in multi-head attention, by default True
        causal : bool, optional
            whether self-attention should be causal, by default False
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_first = norm_first
        self.norm_type = norm_type
        self.bias = bias
        self.causal = causal

        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm1 = nn.LayerNorm(self.d_model) if norm_type == 'layernorm' else RMSNorm(self.d_model)
        self.self_attn = Attention(
            d_model=self.d_model, n_heads=self.n_heads,
            n_kv_heads=None, activation='softmax',
            add_bias_kv=False, add_bias_out=self.bias,
            total_n_heads=None, dropout=self.dropout_rate)
        self.norm2 = nn.LayerNorm(self.d_model) if norm_type == 'layernorm' else RMSNorm(self.d_model)
        self.ff_block = FeedForwardBlock(self.d_model, dff=self.dff, activation=self.activation, use_bias=self.bias)

    def forward(self, x, freqs_cos=None, freqs_sin=None):
        if self.norm_first:
            x = x + self._compute_self_attn(self.norm1(x), freqs_cos=freqs_cos, freqs_sin=freqs_sin)
            x = x + self._apply_ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._compute_self_attn(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin))
            x = self.dropout(x)
            x = self.norm2(x + self._apply_ff_block(x))
        return x

    def _compute_self_attn(self, x, freqs_cos=None, freqs_sin=None):
        x, _ = self.self_attn(query=x, key=x, value=x, is_causal=self.causal,
            need_weights=False, attn_mask=None, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
        x = self.dropout(x)
        return x

    def _apply_ff_block(self, x):
        x = self.ff_block(x)
        x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self,
            d_model: int,
            n_heads: int,
            n_heads_cross: int,
            dff: int,
            dropout_rate: float,
            activation: str,
            norm_first: bool,
            norm_type: str = 'layernorm',
            bias: bool = True,
            causal: bool = False):
        """
        A Transformer Decoder Block.

        Consists of Self-attention, Cross-attention, Feed-forward block and LayerNorms/Residuals.

        Parameters
        ----------
        d_model : int
            model dimension.
        n_heads : int
            number of self-attention heads.
        n_heads_cross : int
            number of cross-attention heads.
        dff : int
            intermediate dimension of feed-forward block.
        dropout_rate : float
            dropout rate.
        activation : str
            name of activation function to use in feed-forward block.
        norm_first : bool
            whether to apply layer normalization before or after attention.
        norm_type: str, optional
            type of normalization to use. 'layernorm' or 'rmsnorm'. Default is 'layernorm'.
        bias : bool, optional
            whether to use bias in multi-head attention, by default True
        causal : bool, optional
            whether self-attention should be causal, by default False
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_heads_cross = n_heads_cross
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_first = norm_first
        self.norm_type = norm_type
        self.bias = bias
        self.causal = causal

        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm1 = nn.LayerNorm(self.d_model) if norm_type == 'layernorm' else RMSNorm(self.d_model)
        self.self_attn = Attention(
            d_model=self.d_model, n_heads=self.n_heads,
            n_kv_heads=None, activation='softmax',
            add_bias_kv=False, add_bias_out=self.bias,
            total_n_heads=None, dropout=self.dropout_rate)
        self.norm2 = nn.LayerNorm(self.d_model) if norm_type == 'layernorm' else RMSNorm(self.d_model)
        self.cross_attn = Attention(
            d_model=self.d_model, n_heads=self.n_heads,
            n_kv_heads=None, activation='softmax',
            add_bias_kv=False, add_bias_out=self.bias,
            total_n_heads=None, dropout=self.dropout_rate)
        self.norm3 = nn.LayerNorm(self.d_model) if norm_type == 'layernorm' else RMSNorm(self.d_model)
        self.ff_block = FeedForwardBlock(self.d_model, dff=self.dff, activation=self.activation, use_bias=self.bias)

    def forward(self, x, context):
        if self.norm_first:
            x = x + self._compute_self_attn(self.norm1(x))
            x = x + self._compute_cross_attn(self.norm2(x), context)
            x = x + self.ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._compute_self_attn(x))
            x = self.norm2(x + self._compute_cross_attn(x, context))
            x = self.norm3(x + self.ff_block(x))
        return x

    def _compute_self_attn(self, x):
        x, _ = self.self_attn(query=x, key=x, value=x, is_causal=self.causal,
            attn_mask=None, need_weights=False, freqs_cos=None, freqs_sin=None)
        x = self.dropout(x)
        return x

    def _compute_cross_attn(self, x, context):
        x, _ = self.cross_attn(query=x, key=context, value=context, is_causal=False,
            attn_mask=None, need_weights=False, freqs_cos=None, freqs_sin=None)

        x = self.dropout(x)
        return x

    def _apply_ff_block(self, x):
        x = self.ff_block(x)
        x = self.dropout(x)
        return x


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
        self.dff = dff if dff is not None else 4 * embed_dim
        self.use_bias = use_bias
        self.activation = activation
        if self.activation != 'swiglu':
            self.activation_ = model_utils.get_activation_function(activation)

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
