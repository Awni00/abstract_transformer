import torch
from torch import nn

import sys; sys.path += ['..', '../..']
from blocks import EncoderBlock, DecoderBlock
from abstract_blocks import AbstractEncoderBlock, AbstractDecoderBlock
from symbol_retrieval import SymbolicAttention, RelationalSymbolicAttention, PositionalSymbolRetriever
from positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEmbeddings
from original_abstractor_module import AbstractorModule


class Seq2SeqAbstractorArcha(nn.Module):
    """Abstractor Seq2Seq Model"""

    def __init__(self,
        input_spec, output_spec, d_model, out_dim,
        n_layers_dec, abstractor_kwargs, decoder_kwargs,
        in_block_size, out_block_size):
        super().__init__()

        self.input_spec = input_spec
        self.output_spec = output_spec
        self.d_model = d_model
        self.out_dim = out_dim
        self.n_layers_dec = n_layers_dec
        self.decoder_kwargs = decoder_kwargs
        self.in_block_size = in_block_size
        self.out_block_size = out_block_size

        # TODO: make positional embedder configurable (learned or fixed sinusoidal, etc)
        if input_spec['type'] == 'token':
            source_embedder = torch.nn.Embedding(input_spec['vocab_size'], d_model)
        elif input_spec['type'] == 'vector':
            source_embedder = torch.nn.Linear(input_spec['dim'], d_model)
        else:
            raise ValueError(f"input_spec['type'] must be 'token' or 'vector', not {input_spec['type']}")

        if output_spec['type'] == 'token':
            target_embedder = torch.nn.Embedding(output_spec['vocab_size'], d_model)
        elif output_spec['type'] == 'vector':
            target_embedder = torch.nn.Linear(output_spec['dim'], d_model)
        else:
            raise ValueError(f"output_spec['type'] must be 'token' or 'vector', not {output_spec['type']}")

        layer_dict = dict(
            source_embedder = source_embedder,
            target_embedder = target_embedder,
            source_pos_embedder = SinusoidalPositionalEncoding(d_model, dropout=0., max_len=in_block_size),
            target_pos_embedder = SinusoidalPositionalEncoding(d_model, dropout=0., max_len=out_block_size),
            # dropout = nn.Dropout(dropout_rate),
            abstractor = AbstractorModule(**abstractor_kwargs),
            decoder_blocks = nn.ModuleList([DecoderBlock(d_model, **decoder_kwargs) for _ in range(n_layers_dec)]),
            final_out = nn.Linear(d_model, out_dim)
        )

        self.layers = nn.ModuleDict(layer_dict)

        # weight-tie target embedder and output layer
        # self.layers.target_embedder.weight = self.layers.final_out.weight

    def forward(self, x, y, targets=None):

        x = self.layers.source_embedder(x)
        y = self.layers.target_embedder(y)
        x = self.layers.source_pos_embedder(x)
        y = self.layers.target_pos_embedder(y)

        x = self.layers.abstractor(x)

        for dec_block in self.layers.decoder_blocks:
            y = dec_block(y, x)

        if targets is not None:
            # compute loss if given targets
            logits = self.layers.final_out(y)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.layers.final_out(y[:, [-1], :])
            loss = None

        return logits, loss

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

class Seq2SeqAbstractorArchb(nn.Module):
    """Abstractor Seq2Seq Model"""

    def __init__(self,
        input_spec, output_spec, d_model, out_dim,
        n_layers_enc, n_layers_dec, encoder_kwargs, abstractor_kwargs, decoder_kwargs,
        in_block_size, out_block_size):
        super().__init__()

        self.input_spec = input_spec
        self.output_spec = output_spec
        self.d_model = d_model
        self.out_dim = out_dim
        self.n_layers_enc = n_layers_enc
        # self.n_layers_abs = n_layers_abs
        self.n_layers_dec = n_layers_dec
        self.encoder_kwargs = encoder_kwargs
        self.decoder_kwargs = decoder_kwargs
        self.in_block_size = in_block_size
        self.out_block_size = out_block_size

        # TODO: make positional embedder configurable (learned or fixed sinusoidal, etc)
        if input_spec['type'] == 'token':
            source_embedder = torch.nn.Embedding(input_spec['vocab_size'], d_model)
        elif input_spec['type'] == 'vector':
            source_embedder = torch.nn.Linear(input_spec['dim'], d_model)
        else:
            raise ValueError(f"input_spec['type'] must be 'token' or 'vector', not {input_spec['type']}")

        if output_spec['type'] == 'token':
            target_embedder = torch.nn.Embedding(output_spec['vocab_size'], d_model)
        elif output_spec['type'] == 'vector':
            target_embedder = torch.nn.Linear(output_spec['dim'], d_model)
        else:
            raise ValueError(f"output_spec['type'] must be 'token' or 'vector', not {output_spec['type']}")

        layer_dict = dict(
            source_embedder = source_embedder,
            target_embedder = target_embedder,
            source_pos_embedder = SinusoidalPositionalEncoding(d_model, dropout=0., max_len=in_block_size),
            target_pos_embedder = SinusoidalPositionalEncoding(d_model, dropout=0., max_len=out_block_size),
            # dropout = nn.Dropout(dropout_rate),
            encoder_blocks = nn.ModuleList([EncoderBlock(d_model, **encoder_kwargs) for _ in range(n_layers_enc)]),
            abstractor = AbstractorModule(**abstractor_kwargs),
            decoder_blocks = nn.ModuleList([DecoderBlock(d_model, **decoder_kwargs) for _ in range(n_layers_dec)]),
            final_out = nn.Linear(d_model, out_dim)
        )

        self.layers = nn.ModuleDict(layer_dict)

        # weight-tie target embedder and output layer
        # self.layers.target_embedder.weight = self.layers.final_out.weight

    def forward(self, x, y, targets=None):

        x = self.layers.source_embedder(x)
        y = self.layers.target_embedder(y)
        x = self.layers.source_pos_embedder(x)
        y = self.layers.target_pos_embedder(y)

        for enc_block in self.layers.encoder_blocks:
            x = enc_block(x)

        x = self.layers.abstractor(x)

        for dec_block in self.layers.decoder_blocks:
            y = dec_block(y, x)

        if targets is not None:
            # compute loss if given targets
            logits = self.layers.final_out(y)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.layers.final_out(y[:, [-1], :])
            loss = None

        return logits, loss

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        # N = self.get_num_params()
        # # 2 * self.n_layers because we have both encoder and decoder
        # H = self.n_heads_enc + self.n_heads_abs # FIXME: this is a hack and probably incorrect!
        # L, Q, T = 2*self.n_layers, self.d_model//self.n_heads_enc, self.block_size
        # flops_per_token = 6*N + 12*L*H*Q*T
        # flops_per_fwdbwd = flops_per_token * T
        # flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # # express our flops throughput as ratio of A100 bfloat16 peak flops
        # flops_achieved = flops_per_iter * (1.0/dt) # per second
        # flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        # mfu = flops_achieved / flops_promised
        # return mfu
        # FIXME: need to implement for encoder-decoder models
        return -1


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
