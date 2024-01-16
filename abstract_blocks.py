import torch
import torch.nn as nn
from multi_head_attention import MultiheadAttention
from blocks import FeedForwardBlock

class AbstractEncoderBlock(nn.Module):

    def __init__(self,
            symbol_retriever: nn.Module,
            d_model: int,
            n_heads_enc: int,
            n_heads_abs: int,
            dff: int,
            dropout_rate: float,
            activation: str,
            norm_first: bool,
            rel_mask_diag: bool = True,
            bias: bool =True,
            causal: bool = False):
        """
        Abstract Encoder Block.

        An Abstract Encoder is a variant of the Transformer Encoder that uses a mixture of specialized attention heads.
        Some attention heads are standard self-attention heads, while other heads are relational cross-attention heads.

        Parameters
        ----------
        symbol_retriever : nn.Module
            symbol retriever module. assigns each object in collection of inputs a symbol from a symbol library.
        d_model : int
            model dimension.
        n_heads_enc : int
            number of standard self-attention heads
        n_heads_abs : int
            number of "abstract" relational cross-attention heads
        dff : int
            intermediate dimension of feed-forward block.
        dropout_rate : float
            dropout rate.
        activation : str
            name of activation function to use in feed-forward block.
        norm_first : bool
            whether to apply layer normalization before or after attention.
        rel_mask_diag : bool, optional
            whether to mask out self-relations in relational cross-attention, by default True
        bias : bool, optional
            whether to use bias in multi-head attention, by default True
        causal : bool, optional
            whether self-attention should be causal, by default False
        """

        super().__init__()
        self.symbol_retriever = symbol_retriever
        self.d_model = d_model
        self.n_heads_enc = n_heads_enc
        self.n_heads_abs = n_heads_abs
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_first = norm_first
        self.rel_mask_diag = rel_mask_diag
        self.bias = bias
        self.causal = causal

        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.self_attn = MultiheadAttention(
            self.d_model, self.n_heads_enc, dropout=self.dropout_rate, bias=self.bias, add_bias_kv=False,
            kdim=self.d_model, vdim=self.d_model, outdim=self.d_model//2, batch_first=True)
        self.rel_attn = MultiheadAttention(
            self.d_model, self.n_heads_abs, dropout=self.dropout_rate, bias=self.bias, add_bias_kv=False,
            kdim=self.d_model, vdim=self.d_model, outdim=self.d_model//2, batch_first=True)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.ff_block = FeedForwardBlock(self.d_model, self.dff, self.bias, self.activation)

    def forward(self, x):
        if self.norm_first:
            x = x + self._compute_abstract_attn(self.norm1(x))
            x = x + self._apply_ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._compute_abstract_attn(x))
            x = self.dropout(x)
            x = self.norm2(x + self._apply_ff_block(x))
        return x

    def _compute_abstract_attn(self, x):
        # retrieve symbols
        symbols = self.symbol_retriever(x)

        # compute standard self-attention
        self_attn_mask = self._compute_self_attn_mask(x)
        E = self.self_attn(query=x, key=x, value=x, need_weights=False, attn_mask = self_attn_mask,  is_causal=self.causal)[0]

        # compute relational cross-attention
        rel_attn_mask = self._compute_rel_attn_mask(x)
        A = self.rel_attn(query=x, key=x, value=symbols, need_weights=False, is_causal=False, attn_mask=rel_attn_mask)[0]

        # concat E and A
        x = torch.concat((E, A), dim=-1)

        x = self.dropout(x) # dropout

        return x

    def _apply_ff_block(self, x):
        x = self.ff_block(x)
        x = self.dropout(x)
        return x

    def _compute_rel_attn_mask(self, size):
        mask = torch.zeros(size=(size, size))
        if self.rel_mask_diag:
            mask += compute_diag_mask(size)
        if self.causal:
            mask += torch.nn.modules.transformer.Transformer.generate_square_subsequent_mask(size)

        # edge case: if both diagonal mask and causal mask, the all elements of the first row will be -inf
        # this becomes NaN after softmax, so we set it to 0
        if self.rel_mask_diag and self.causal:
            mask[0,0] = 0.0

        if self.rel_mask_diag or self.causal:
            return mask
        else:
            return None

    def _compute_self_attn_mask(self, x):
        if self.causal:
            causal_mask = torch.nn.modules.transformer.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            return causal_mask
        else:
            return None

class AbstractDecoderBlock(nn.Module):
    def __init__(self,
                symbol_retriever: nn.Module,
                d_model: int,
                n_heads_enc: int,
                n_heads_abs: int,
                n_heads_cross: int,
                dff: int,
                dropout_rate: float,
                activation: str,
                norm_first: bool,
                rel_mask_diag: bool = True,
                bias: bool = True,
                causal: bool = True):
        """
        Abstract Decoder Block.

        An Abstract Decoder is a variant of the Transformer Decoder that uses a mixture of specialized attention heads.
        Some attention heads are standard self-attention heads, while other heads are relational cross-attention heads.

        Parameters
        ----------
        symbol_retriever : nn.Module
            symbol retriever module. assigns each object in collection of inputs a symbol from a symbol library.
        d_model : int
            model dimension.
        n_heads_enc : int
            number of standard self-attention heads.
        n_heads_abs : int
            number of "abstract" relational cross-attention heads.
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
        rel_mask_diag : bool, optional
            whether to mask out self-relations in relational cross-attention, by default True
        bias : bool, optional
            whether to use bias in multi-head attention, by default True
        causal : bool, optional
            whether self-attention should be causal, by default False
        """

        super().__init__()
        self.symbol_retriever = symbol_retriever
        self.d_model = d_model
        self.n_heads_enc = n_heads_enc
        self.n_heads_abs = n_heads_abs
        self.n_heads_cross = n_heads_cross
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_first = norm_first
        self.rel_mask_diag = rel_mask_diag
        self.bias = bias
        self.causal = causal

        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.self_attn = MultiheadAttention(
            self.d_model, self.n_heads_enc, dropout=self.dropout_rate, bias=self.bias, add_bias_kv=False,
            kdim=self.d_model, vdim=self.d_model, outdim=self.d_model//2, batch_first=True)
        self.rel_attn = MultiheadAttention(
            self.d_model, self.n_heads_abs, dropout=self.dropout_rate, bias=self.bias, add_bias_kv=False,
            kdim=self.d_model, vdim=self.d_model, outdim=self.d_model//2, batch_first=True)

        self.norm2 = nn.LayerNorm(self.d_model)
        self.cross_attn = MultiheadAttention(
            self.d_model, self.n_heads_cross, dropout=self.dropout_rate, bias=self.bias, add_bias_kv=False, kdim=self.d_model, vdim=self.d_model, batch_first=True)
        self.norm3 = nn.LayerNorm(self.d_model)
        self.ff_block = FeedForwardBlock(self.d_model, self.dff, self.bias, self.activation)

    def forward(self, x, context):
        if self.norm_first:
            x = x + self._compute_abstract_attn(self.norm1(x))
            x = x + self._compute_cross_attn(self.norm2(x), context)
            x = x + self.ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._compute_abstract_attn(x))
            x = self.norm2(x + self._compute_cross_attn(x, context))
            x = self.norm3(x + self.ff_block(x))
        return x

    def _compute_abstract_attn(self, x):
        # retrieve symbols
        symbols = self.symbol_retriever(x)

        # compute standard self-attention
        self_attn_mask = self._compute_self_attn_mask(x)
        E = self.self_attn(query=x, key=x, value=x, need_weights=False, attn_mask = self_attn_mask,  is_causal=self.causal)[0]

        # compute relational cross-attention
        rel_attn_mask = self._compute_rel_attn_mask(x)
        A = self.rel_attn(query=x, key=x, value=symbols, need_weights=False, is_causal=False, attn_mask=rel_attn_mask)[0]

        # concat E and A
        x = torch.concat((E, A), dim=-1)

        x = self.dropout(x) # dropout

        return x

    def _compute_cross_attn(self, x, context):
        x = self.cross_attn(query=x, key=context, value=context, need_weights=False, is_causal=False)[0]
        x = self.dropout(x)
        return x

    def _compute_rel_attn_mask(self, size):
        mask = torch.zeros(size=(size, size))
        if self.rel_mask_diag:
            mask += compute_diag_mask(size)
        if self.causal:
            mask += torch.nn.modules.transformer.Transformer.generate_square_subsequent_mask(size)

        # edge case: if both diagonal mask and causal mask, the all elements of the first row will be -inf
        # this becomes NaN after softmax, so we set it to 0
        if self.rel_mask_diag and self.causal:
            mask[0,0] = 0.0

        if self.rel_mask_diag or self.causal:
            return mask
        else:
            return None

    def _compute_self_attn_mask(self, x):
        if self.causal:
            causal_mask = torch.nn.modules.transformer.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            return causal_mask
        else:
            return None

    def _apply_ff_block(self, x):
        x = self.ff_block(x)
        x = self.dropout(x)
        return x

def compute_diag_mask(size, device=None):
    """computes an attention mask with -inf on the diagonal and 0 elsewhere"""

    diag_mask = torch.eye(size, device=device)
    diag_mask = diag_mask.masked_fill(diag_mask == 1, float('-inf'))
    return diag_mask