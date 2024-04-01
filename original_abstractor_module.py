import torch
from torch import nn
from multi_head_attention import MultiheadAttention
from transformer_blocks import FeedForwardBlock
from symbol_retrieval import PositionalSymbolRetriever, SymbolicAttention
from positional_encoding import SinusoidalPositionalEncoding

class AbstractorModule(nn.Module):
    """An implementation of the original Abstractor module."""
    def __init__(self,
            n_layers: int,
            d_model: int,
            n_heads: int,
            dff: int,
            use_self_attn: bool,
            dropout_rate: float,
            activation: str,
            norm_first: bool,
            symbol_retriever_type: str,
            symbol_retriever_kwargs: dict,
            symbol_add_pos_embedding: bool,
            max_len: int,
            bias: bool = True):

        super(AbstractorModule, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_first = norm_first
        self.symbol_retriever_type = symbol_retriever_type
        self.symbol_add_pos_embedding = symbol_add_pos_embedding
        self.max_len = max_len
        self.bias = bias

        if self.symbol_retriever_type == 'positional':
            self.symbol_retriever = PositionalSymbolRetriever(**symbol_retriever_kwargs)
        elif self.symbol_retriever_type == 'symbolic_attention':
            self.symbol_retriever = SymbolicAttention(**symbol_retriever_kwargs)
        else:
            raise ValueError("invalid symbol_retriever_type.")


        self.abstractor_layers = nn.ModuleList([
            AbstractorModuleLayer(
                d_model=d_model, n_heads=n_heads, dff=dff, dropout_rate=dropout_rate,
                activation=activation, norm_first=norm_first, use_self_attn=use_self_attn, bias=bias)
            for _ in range(n_layers)
            ])

        if self.symbol_add_pos_embedding:
            self.add_pos_embedding = SinusoidalPositionalEncoding(
                d_model, dropout=dropout_rate, max_len=max_len)

    def forward(self, x):

        s = self.symbol_retriever(x)

        if self.symbol_add_pos_embedding:
            s = self.add_pos_embedding(s)

        a = s
        for abstractor_layer in self.abstractor_layers:
            a = abstractor_layer(x, a)

        return a

class AbstractorModuleLayer(nn.Module):
    def __init__(self,
            d_model: int,
            n_heads: int,
            dff: int,
            dropout_rate: float,
            activation: str,
            norm_first: bool,
            use_self_attn: bool = True,
            bias: bool = True):

        super(AbstractorModuleLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_first = norm_first
        self.use_self_attn = use_self_attn
        self.bias = bias

        self.rel_attn = MultiheadAttention(
            self.d_model, self.n_heads, dropout=self.dropout_rate, bias=self.bias, add_bias_kv=False,
            kdim=self.d_model, vdim=self.d_model, outdim=self.d_model, batch_first=True)
        if self.use_self_attn:
            self.self_attn = MultiheadAttention(
                self.d_model, self.n_heads, dropout=self.dropout_rate, bias=self.bias, add_bias_kv=False,
                kdim=self.d_model, batch_first=True
            )
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.ff_block = FeedForwardBlock(self.d_model, self.dff, self.bias, self.activation)

    def forward(self, x, s):
        if self.use_self_attn:
            s = self._apply_self_attn(s)
        a = self._apply_rel_attn(x, s)
        a = self._apply_ff_block(a)

        return  a

    def _apply_rel_attn(self, x, s):
        if self.norm_first:
            x_ = self.norm1(x)
            x = x + self.rel_attn(query=x_, key=x_, value=s, need_weights=False)[0]
        else:
            x = self.norm1(x + self.rel_attn(query=x, key=x, value=s, need_weights=False)[0])
            x = self.dropout(x)
        return x

    def _apply_self_attn(self, x):
        if self.norm_first:
            x_ = self.norm1(x)
            x = x + self.self_attn(query=x_, key=x_, value=x_, need_weights=False)[0]
        else:
            x = self.norm1(x + self.self_attn(query=x, key=x, value=x, need_weights=False)[0])
            x = self.dropout(x)
        return x

    def _apply_ff_block(self, x):
        if self.norm_first:
            x_ = self.norm2(x)
            ff_out = self.ff_block(x_)
            ff_out = self.dropout(ff_out)
            x = x + ff_out
        else:
            x_ = x
            ff_out = self.ff_block(x_)
            ff_out = self.dropout(ff_out)
            x = self.norm2(x + ff_out)
        return x