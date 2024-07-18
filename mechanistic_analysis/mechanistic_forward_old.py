import torch

def symbolic_attn_forward_get_weights(mod, x):
    '''a variant of the forward call for symbolic attention that returns the attention weights'''

    mod.eval()

    batch_size, seq_len, dim = x.size()

    # create query from input
    query = mod.q_proj(x)
    query = query.view(batch_size, seq_len, mod.n_heads, dim // mod.n_heads).transpose(1, 2)

    # create keys from template features
    key = mod.template_features.view(mod.n_symbols, mod.n_heads, mod.d_model // mod.n_heads).transpose(0, 1)
    key = mod._repeat_kv(key, batch_size)

    # create values from symbol library
    value = mod.symbol_library.view(mod.n_symbols, mod.n_heads, mod.d_model // mod.n_heads).transpose(0, 1)
    value = mod._repeat_kv(value, batch_size)

    # retrieved_symbols = torch.nn.functional.scaled_dot_product_attention(
    #     query, key, value,
    #     scale=self.scale, dropout_p=self.dropout, attn_mask=None, is_causal=False)
    scale = mod.scale if mod.scale is not None else (mod.d_model/mod.n_heads) ** -0.5

    attn_scores = torch.matmul(query, key.transpose(2, 3)) * scale
    attn_scores = torch.nn.functional.softmax(attn_scores, dim=-1)
    retrieved_symbols = torch.matmul(attn_scores, value)

    retrieved_symbols = retrieved_symbols.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)

    return retrieved_symbols, attn_scores

def block_forward_get_weights(mod, x, symbols, freqs_cos=None, freqs_sin=None):
    '''a variant of the forward call for a block that returns the attention weights'''
    mod.eval()

    def _compute_dual_attn(mod, x, symbols, freqs_cos=None, freqs_sin=None):

        x, sa_attn_scores, (ra_attn_scores, ra_rels) = mod.dual_attn(x, symbols,
            need_weights=True, is_causal=mod.causal,
            freqs_cos=freqs_cos, freqs_sin=freqs_sin)

        x = mod.dropout(x) # dropout

        return x, sa_attn_scores, ra_attn_scores, ra_rels

    if mod.norm_first:
        attn_out, sa_attn_scores, ra_attn_scores, ra_rels = _compute_dual_attn(mod, mod.norm1(x), symbols, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
        x = x + attn_out
        x = x + mod._apply_ff_block(mod.norm2(x))
    else:
        attn_out, sa_attn_scores, ra_attn_scores = _compute_dual_attn(mod, x, symbols, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
        x = mod.norm1(x + attn_out)
        x = mod.dropout(x)
        x = mod.norm2(x + mod._apply_ff_block(x))

    return x, sa_attn_scores, ra_attn_scores, ra_rels

def datlm_forward_w_intermediate_results(model, x):
    model.eval()

    device = x.device
    b, t = x.size()
    assert t <= model.block_size, f'Input sequence length {t} exceeds maximum block size {model.block_size}'

    x = model.layers.token_embedder(x)

    if model.pos_enc_type == 'pos_emb':
        # if using positional embeddings, add positional embeeddings
        positions = torch.arange(0, t, dtype=torch.long, device=device)
        positional_embedding = model.layers.positional_embedder(positions)
        x = x + positional_embedding
        freqs_cos, freqs_sin = None, None # not using RoPE
    elif model.pos_enc_type == 'RoPE':
        # otherwise, get the RoPE matrices
        freqs_cos = model.freqs_cos[:t]
        freqs_sin = model.freqs_sin[:t]


    intermediate_results = dict(symbol_attn_scores = [], sa_attn_scores = [], ra_attn_scores = [], ra_rels = [])


    for block in model.layers.blocks:
        symbols, symbol_attn_scores_ = symbolic_attn_forward_get_weights(model.layers.symbol_retriever, x)
        intermediate_results['symbol_attn_scores'].append(symbol_attn_scores_)

        x, sa_attn_scores, ra_attn_scores, ra_rels = block_forward_get_weights(block, x, symbols, freqs_cos=freqs_cos, freqs_sin=freqs_sin)

        intermediate_results['sa_attn_scores'].append(sa_attn_scores)
        intermediate_results['ra_attn_scores'].append(ra_attn_scores)
        intermediate_results['ra_rels'].append(ra_rels)
        # symbols = model.layers.symbol_retriever(x)
        # x = block(x, symbols, freqs_cos=freqs_cos, freqs_sin=freqs_sin)

    x = model.layers.norm(x)

    logits = model.layers.final_out(x)

    return logits, intermediate_results