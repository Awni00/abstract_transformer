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
        self.bias = bias
        self.causal = causal

        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.self_attn = Attention(
            d_model=self.d_model, n_heads=self.n_heads,
            n_kv_heads=None, activation='softmax',
            add_bias_kv=False, add_bias_out=self.bias,
            total_n_heads=None, dropout=self.dropout_rate)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.ff_block = FeedForwardBlock(self.d_model, self.dff, self.bias, self.activation)

    def forward(self, x):
        if self.norm_first:
            x = x + self._compute_self_attn(self.norm1(x))
            x = x + self._apply_ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._compute_self_attn(x))
            x = self.dropout(x)
            x = self.norm2(x + self._apply_ff_block(x))
        return x

    def _compute_self_attn(self, x):
        x, _ = self.self_attn(query=x, key=x, value=x, is_causal=self.causal,
            need_weights=False, attn_mask=None, freqs_cos=None, freqs_sin=None)
        x = self.dropout(x)
        return x

    def _apply_ff_block(self, x):
        x = self.ff_block(x)
        x = self.dropout(x)
        return x



class DecoderBlock(nn.Module):

    def __init__(self,
            d_model,
            n_heads,
            n_heads_cross,
            dff,
            dropout_rate,
            activation,
            norm_first,
            bias=True,
            causal=True):
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
        self.bias = bias
        self.causal = causal

        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.self_attn = Attention(
            d_model=self.d_model, n_heads=self.n_heads,
            n_kv_heads=None, activation='softmax',
            add_bias_kv=False, add_bias_out=self.bias,
            total_n_heads=None, dropout=self.dropout_rate)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.cross_attn = Attention(
            d_model=self.d_model, n_heads=self.n_heads,
            n_kv_heads=None, activation='softmax',
            add_bias_kv=False, add_bias_out=self.bias,
            total_n_heads=None, dropout=self.dropout_rate)
        self.norm3 = nn.LayerNorm(self.d_model)
        self.ff_block = FeedForwardBlock(self.d_model, self.dff, self.bias, self.activation)

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
        x = self.cross_attn(query=x, key=context, value=context, is_causal=False,
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
            use_bias: bool = False,
            activation: str = 'relu'):
        """
        Feed-forward block.

        A 2-layer neural network with activation function in between.

        Parameters
        ----------
        embed_dim : int
            embedding dimension of input.
        dff : int, optional
            size of intermediate layer. if None, 4 * embed_dim.
        use_bias : bool, optional
            whether to use bias in linear layers, by default False
        activation : str, optional
            name of activation function, by default 'relu'
        """

        super().__init__()
        self.embed_dim = embed_dim
        self.dff = dff if dff is not None else 4 * embed_dim
        self.use_bias = use_bias

        self.linear1 = nn.Linear(self.embed_dim, self.dff, bias=self.use_bias)
        self.activation = model_utils.get_activation_function(activation)
        self.linear2 = nn.Linear(self.dff, self.embed_dim, bias=self.use_bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


