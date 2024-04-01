import torch
from torch import nn
from transformer_blocks import EncoderBlock
from abstract_blocks import AbstractEncoderBlock
from symbol_retrieval import SymbolicAttention, RelationalSymbolicAttention, PositionalSymbolRetriever

class TransformerLM(nn.Module):
    """Transformer Language Model"""

    def __init__(self, vocab_size, d_model, n_layers, n_heads, dff, dropout_rate, activation, norm_first, max_block_size, bias=True):
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

        self.layers = nn.ModuleDict(dict(
            token_embedder = nn.Embedding(vocab_size, d_model),
            positional_embedder = nn.Embedding(max_block_size, d_model),
            dropout = nn.Dropout(dropout_rate),
            blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, dff, dropout_rate,
                activation, norm_first, bias, causal=True) for _ in range(n_layers)]),
            final_out = nn.Linear(d_model, vocab_size)
        ))

        # weight-tying embedder and final layer
        self.layers.token_embedder.weights = self.layers.final_out


    def forward(self, x, targets=None):
        device = x.device
        b, t = x.size()
        assert t <= self.block_size, f'Input sequence length {t} exceeds maximum block size {self.block_size}'

        token_embedding = self.layers.token_embedder(x)
        positions = torch.arange(0, t, dtype=torch.long, device=device)
        positional_embedding = self.layers.positional_embedder(positions)
        x = token_embedding + positional_embedding

        for enc_block in self.layers.blocks:
            x = enc_block(x)

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
        if non_embedding:
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

class AbstractTransformerLM(nn.Module):
    """Abstract Transformer Language Model"""
    def __init__(self,
            vocab_size,
            d_model,
            n_layers,
            n_heads_enc,
            n_heads_abs,
            symbol_retrieval_kwargs,
            dff,
            dropout_rate,
            activation,
            norm_first,
            max_block_size,
            symbol_retrieval='sym_attn',
            bias=True):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads_enc = n_heads_enc
        self.n_heads_abs = n_heads_abs
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_first = norm_first
        self.block_size = max_block_size
        self.bias = bias

        if symbol_retrieval == 'sym_attn':
            self.symbol_retriever = SymbolicAttention(**symbol_retrieval_kwargs)
        elif symbol_retrieval == 'rel_sym_attn':
            self.symbol_retriever = RelationalSymbolicAttention(**symbol_retrieval_kwargs)
        elif symbol_retrieval == 'pos_sym_retriever':
            self.symbol_retriever = PositionalSymbolRetriever(**symbol_retrieval_kwargs)
        else:
            raise ValueError(f"`symbol_retrieval` must be one of 'sym_attn', 'rel_sym_attn', or 'pos_sym_retriever'. received {symbol_retrieval}")

        self.layers = nn.ModuleDict(dict(
            token_embedder = nn.Embedding(vocab_size, d_model),
            positional_embedder = nn.Embedding(max_block_size, d_model),
            dropout = nn.Dropout(dropout_rate),

            blocks = nn.ModuleList([AbstractEncoderBlock(self.symbol_retriever, d_model, n_heads_enc, n_heads_abs, dff, dropout_rate,
                activation, norm_first, bias, causal=True) for _ in range(n_layers)]),
            final_out = nn.Linear(d_model, vocab_size)
        ))

        # weight-tying embedder and final layer
        self.layers.token_embedder.weights = self.layers.final_out

    def forward(self, x, targets=None):
        device = x.device
        b, t = x.size()
        assert t <= self.block_size, f'Input sequence length {t} exceeds maximum block size {self.block_size}'

        token_embedding = self.layers.token_embedder(x)
        positions = torch.arange(0, t, dtype=torch.long, device=device)
        positional_embedding = self.layers.positional_embedder(positions)
        x = token_embedding + positional_embedding

        for enc_block in self.layers.blocks:
            x = enc_block(x)

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
        print('WARNING: estimate_mfu implementation not checked for AbstractTransformerLM')
        N = self.get_num_params()
        nh = self.n_heads_enc + self.n_heads_abs
        L, H, Q, T = self.n_layers, nh, self.d_model//self.n_heads_enc, self.block_size
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
        if non_embedding:
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