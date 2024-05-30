import torch
from torch import nn
from transformer_blocks import EncoderBlock
from dual_attn_blocks import DualAttnEncoderBlock
from symbol_retrieval import SymbolicAttention, RelationalSymbolicAttention, PositionalSymbolRetriever, PositionRelativeSymbolRetriever
from attention_utils import precompute_freqs_cis

class TransformerLM(nn.Module):
    """Transformer Language Model"""

    def __init__(self,
        vocab_size,
        d_model,
        n_layers,
        n_heads,
        dff,
        dropout_rate,
        activation,
        norm_first,
        max_block_size,
        bias=True,
        pos_enc_type='pos_emb'
        ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_first = norm_first
        self.block_size = max_block_size
        self.bias = bias
        self.pos_enc_type = pos_enc_type

        layers = dict(
            token_embedder = nn.Embedding(vocab_size, d_model),
            dropout = nn.Dropout(dropout_rate),
            blocks = nn.ModuleList([EncoderBlock(d_model=d_model, n_heads=n_heads, dff=dff, dropout_rate=dropout_rate,
                activation=activation, norm_first=norm_first, bias=bias, causal=True) for _ in range(n_layers)]),
            final_out = nn.Linear(d_model, vocab_size)
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
        self.layers.token_embedder.weights = self.layers.final_out

        # NOTE: GPT2 paper suggests special scaled init to the residual projections

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

        for enc_block in self.layers.blocks:
            x = enc_block(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin)

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
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311

        N = self.get_num_params()
        L, H, Q, T = self.n_layers, self.n_heads, self.d_model//self.n_heads, self.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
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
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

class DualAttnTransformerLM(nn.Module):
    """Dual Attention Transformer Language Model"""
    def __init__(self,
            vocab_size,
            d_model,
            n_layers,
            n_heads_sa,
            n_heads_ra,
            symbol_retrieval_kwargs,
            dff,
            dropout_rate,
            activation,
            norm_first,
            max_block_size,
            sa_kwargs: dict = None,
            ra_kwargs: dict = None,
            ra_type: str = 'relational_attention',
            symbol_retrieval='symbolic_attention',
            pos_enc_type='pos_emb',
            bias=True):
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
        self.block_size = max_block_size
        self.ra_type = ra_type
        self.symbol_retriever = symbol_retrieval
        self.pos_enc_type = pos_enc_type
        self.bias = bias

        self.n_heads = n_heads_sa + n_heads_ra

        if symbol_retrieval == 'symbolic_attention':
            symbol_retriever = SymbolicAttention(**symbol_retrieval_kwargs)
        elif symbol_retrieval == 'rel_sym_attn':
            symbol_retriever = RelationalSymbolicAttention(**symbol_retrieval_kwargs)
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
                activation=activation, norm_first=norm_first, sa_kwargs=sa_kwargs, ra_kwargs=ra_kwargs, ra_type=ra_type, causal=True)
                for _ in range(n_layers)]),
            final_out = nn.Linear(d_model, vocab_size)
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
        self.layers.token_embedder.weights = self.layers.final_out

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
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

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
