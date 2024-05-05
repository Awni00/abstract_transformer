"""Implementation of Vision Transformer (ViT) and Vision Abstract Transformer (VAT)"""

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from abstract_blocks import AbstractEncoderBlock
from symbol_retrieval import (
    PositionalSymbolRetriever, PositionRelativeSymbolRetriever, RelationalSymbolicAttention, SymbolicAttention)
from transformer_blocks import EncoderBlock


class ViT(nn.Module):
    """Vision Transformer"""
    def __init__(self,
        image_shape,
        patch_size,
        num_classes,
        d_model,
        n_layers,
        n_heads,
        dff,
        dropout_rate,
        activation,
        norm_first,
        norm_type='layernorm',
        bias=True,
        pool='cls'):

        super(ViT, self).__init__()
        self.img_channels, self.img_width, self.img_height = image_shape
        self.patch_width, self.patch_height = patch_size
        self.num_classes = num_classes

        self.num_patches = (self.img_width // self.patch_width) * (self.img_height // self.patch_height)

        self.patch_dim = self.patch_width * self.patch_height * self.img_channels

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool # type of pooling

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_first = norm_first
        self.norm_type = norm_type
        self.bias = bias

        # extract patches from image and apply linear map
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.dropout = nn.Dropout(self.dropout_rate)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model=d_model, n_heads=n_heads, dff=dff, dropout_rate=dropout_rate,
                activation=activation, norm_first=norm_first, norm_type=norm_type, bias=bias, causal=False) for _ in range(n_layers)])

        self.final_out = nn.Linear(self.d_model, self.num_classes)

    def forward(self, x):

        # extract patches and apply linear map
        x = self.to_patch_embedding(x)
        bsz, n, _ = x.shape

        # repeat class token across batch
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = bsz)
        # prepend class token to input
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional embedding to all tokens
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # pass through transformer
        for block in self.encoder_blocks:
            x = block(x)

        # pool tokens
        if self.pool == 'cls':
            x = x[:, 0]
        elif self.pool == 'mean':
            # NOTE: if mean-pooling, do we need class token?
            x = x.mean(dim=1)

        x = self.final_out(x)

        return x


class VAT(nn.Module):
    """Vision Abstract Transformer"""
    def __init__(self,
        image_shape,
        patch_size,
        num_classes,
        d_model,
        n_layers,
        n_heads_sa,
        n_heads_rca,
        dff,
        dropout_rate,
        activation,
        norm_first,
        symbol_retrieval,
        symbol_retrieval_kwargs,
        rca_type='standard',
        rca_kwargs=None,
        bias=True,
        norm_type='layernorm',
        pool='cls'):

        super(VAT, self).__init__()
        self.img_channels, self.img_width, self.img_height = image_shape
        self.patch_width, self.patch_height = patch_size
        self.num_classes = num_classes

        self.num_patches = (self.img_width // self.patch_width) * (self.img_height // self.patch_height)

        self.patch_dim = self.patch_width * self.patch_height * self.img_channels

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool # type of pooling

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads_sa = n_heads_sa
        self.n_heads_rca = n_heads_rca
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_first = norm_first
        self.norm_type = norm_type
        self.bias = bias
        self.rca_type = rca_type
        self.rca_kwargs = rca_kwargs if rca_kwargs is not None else {}
        self.symbol_retrieval = symbol_retrieval

        if symbol_retrieval == 'sym_attn':
            self.symbol_retriever = SymbolicAttention(**symbol_retrieval_kwargs)
        elif symbol_retrieval == 'rel_sym_attn':
            self.symbol_retriever = RelationalSymbolicAttention(**symbol_retrieval_kwargs)
        elif symbol_retrieval == 'pos_sym_retriever':
            self.symbol_retriever = PositionalSymbolRetriever(**symbol_retrieval_kwargs)
        elif symbol_retrieval == 'pos_relative':
            self.symbol_retriever = PositionRelativeSymbolRetriever(**symbol_retrieval_kwargs)
            # NOTE: pos_relativie symbols may not make too much sense for ViT-type models since positions encode 2 dimensions
        else:
            raise ValueError(
                f"`symbol_retrieval` must be one of 'sym_attn', 'rel_sym_attn', 'pos_sym_retriever' or 'pos_relative."
                f"received {symbol_retrieval}")


        # extract patches from image and apply linear map
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.dropout = nn.Dropout(self.dropout_rate)

        self.encoder_blocks = nn.ModuleList([AbstractEncoderBlock(
            d_model=d_model, n_heads_sa=n_heads_sa, n_heads_rca=n_heads_rca, dff=dff, dropout_rate=dropout_rate,
            activation=activation, norm_first=norm_first, norm_type=norm_type, bias=bias, causal=False,
            rca_type=self.rca_type, rca_kwargs=self.rca_kwargs)
            for _ in range(n_layers)])

        self.final_out = nn.Linear(self.d_model, self.num_classes)

    def forward(self, x):

        # extract patches and apply linear map
        x = self.to_patch_embedding(x)
        bsz, n, _ = x.shape

        # repeat class token across batch
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = bsz)
        # prepend class token to input
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional embedding to all tokens
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # pass through transformer
        for block in self.encoder_blocks:
            symbols = self.symbol_retriever(x)
            x = block(x, symbols)

        # pool tokens
        if self.pool == 'cls':
            x = x[:, 0]
        elif self.pool == 'mean':
            # NOTE: if mean-pooling, do we need class token?
            x = x.mean(dim=1)

        x = self.final_out(x)

        return x

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
