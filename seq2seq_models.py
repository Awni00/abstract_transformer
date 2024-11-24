import torch
from torch import nn

from transformer_blocks import EncoderBlock, DecoderBlock
from dual_attn_blocks import DualAttnEncoderBlock, DualAttnDecoderBlock
from symbol_retrieval import SymbolicAttention, RelationalSymbolicAttention, PositionalSymbolRetriever, PositionRelativeSymbolRetriever
from positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEmbeddings
from attention_utils import precompute_freqs_cis

class Seq2SeqTransformer(nn.Module):
    """Transformer Language Model"""

    def __init__(self,
        input_spec: dict,
        output_spec: dict,
        d_model: int,
        out_dim: int,
        n_layers_enc: int,
        n_layers_dec: int,
        encoder_kwargs: dict,
        decoder_kwargs: dict,
        in_block_size: int,
        out_block_size: int,
        pos_enc_type = 'sinusoidal', # 'sinusoidal' or 'learned' or 'RoPE
        tie_weights: bool = True,
        loss_ignore_idx: int = -1):
        """Seq2Seq Encoder-Decoder Transformer.

        Parameters
        ----------
        input_spec : dict
            description of input format. dictionary with key 'type' with values 'token' or 'vector'.
            if 'token', must also have 'vocab_size'. if 'vector', must also have 'dim'.
        output_spec : dict
            description of output format. dictionary with key 'type' with values 'token' or 'vector'.
            if 'token', must also have 'vocab_size'. if 'vector', must also have 'dim'.
        d_model : int
            model dimension.
        out_dim : int
            output dimension (e.g., output vocab size)
        n_layers_enc : int
            number of encoder layers.
        n_layers_dec : int
            number of decoder layers.
        encoder_kwargs : dict
            keyword arguments for encoder blocks.
        decoder_kwargs : dict
            keyword arguments for decoder blocks.
        in_block_size : int
            block size for input sequence.
        out_block_size : int
            block size for target sequence.
        pos_enc_type : str, optional
            type of positional encoding to use. must be one of 'sinusoidal', 'learned', or 'RoPE, by default 'sinusoidal'.
        tie_weights : bool, optional
            whether to tie weights between target embedder and final layer weights, by default True
        loss_ignore_idx : int, optional
            idx of class to ignore when computing loss, by default -1
        """
        super().__init__()

        self.input_spec = input_spec
        self.output_spec = output_spec
        self.d_model = d_model
        self.out_dim = out_dim
        self.n_layers_enc = n_layers_enc
        self.n_layers_dec = n_layers_dec
        self.pos_enc_type = pos_enc_type
        self.encoder_kwargs = encoder_kwargs
        self.decoder_kwargs = decoder_kwargs
        self.in_block_size = in_block_size
        self.out_block_size = out_block_size
        self.loss_ignore_idx = loss_ignore_idx
        self.n_heads = encoder_kwargs['n_heads'] # assume same number of heads for encoder and decoder
        self.d_head = d_model // self.n_heads

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
            # dropout = nn.Dropout(dropout_rate),
            encoder_blocks = nn.ModuleList([EncoderBlock(d_model, **encoder_kwargs) for _ in range(n_layers_enc)]),
            decoder_blocks = nn.ModuleList([DecoderBlock(d_model, **decoder_kwargs) for _ in range(n_layers_dec)]),
            final_out = nn.Linear(d_model, out_dim)
        )
        if pos_enc_type == 'sinusoidal':
            layer_dict['source_pos_embedder'] = SinusoidalPositionalEncoding(d_model, dropout=0., max_len=in_block_size)
            layer_dict['target_pos_embedder'] = SinusoidalPositionalEncoding(d_model, dropout=0., max_len=out_block_size)
        elif pos_enc_type == 'learned':
            layer_dict['source_pos_embedder'] = LearnedPositionalEmbeddings(d_model, max_len=in_block_size)
            layer_dict['target_pos_embedder'] = LearnedPositionalEmbeddings(d_model, max_len=out_block_size)
        elif pos_enc_type == 'RoPE':
            # if using RoPE, precompute RoPE sine-cosine rotation matrices
            freqs_cos, freqs_sin = precompute_freqs_cis(self.d_head, max(self.in_block_size, self.out_block_size))
            self.register_buffer("freqs_cos", freqs_cos, persistent=False)
            self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        else:
            raise ValueError('`pos_enc_type` invalid')

        self.layers = nn.ModuleDict(layer_dict)

        # weight-tying embedder and final layer
        if tie_weights:
            self.layers.target_embedder.weights = self.layers.final_out



    def forward(self, x, y, targets=None):

        x = self.layers.source_embedder(x)
        y = self.layers.target_embedder(y)

        if self.pos_enc_type == 'sinusoidal' or self.pos_enc_type == 'learned':
            # if using positional embeddings, add positional embeeddings
            x = self.layers.source_pos_embedder(x)
            y = self.layers.target_pos_embedder(y)
            freqs_cos_x, freqs_sin_x, freqs_cos_y, freqs_sin_y = None, None, None, None # not using RoPE
        elif self.pos_enc_type == 'RoPE':
            # otherwise, get the RoPE matrices
            tx, ty = x.size(1), y.size(1)
            freqs_cos_x = self.freqs_cos[:tx]
            freqs_sin_x = self.freqs_sin[:tx]
            freqs_cos_y = self.freqs_cos[:ty]
            freqs_sin_y = self.freqs_sin[:ty]

        for enc_block in self.layers.encoder_blocks:
            x = enc_block(x, freqs_cos=freqs_cos_x, freqs_sin=freqs_sin_x)

        for dec_block in self.layers.decoder_blocks:
            y = dec_block(y, x, freqs_cos=freqs_cos_y, freqs_sin=freqs_sin_y)

        if targets is not None:
            # compute loss if given targets
            logits = self.layers.final_out(y)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=self.loss_ignore_idx)
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
        # NOTE: Model Flops Utilization (MFU) is a measure of how much of the peak FLOPS of the GPU is being utilized.
        # PaLM paper has computed this for standard Transformers
        # haven't done this yet for encoder-decoder architectures, so this is a placeholder

        return -1.0

class Seq2SeqDualAttnTransformer(nn.Module):
    """Dual Attention Transformer Seq2Seq Model"""

    def __init__(self,
        input_spec: dict,
        output_spec: dict,
        symbol_retrieval: str,
        symbol_retrieval_kwargs: dict,
        d_model: int,
        out_dim: int,
        n_layers_enc: int,
        n_layers_dec: int,
        encoder_kwargs: dict,
        decoder_kwargs: dict,
        in_block_size: int,
        out_block_size: int,
        update_symbols_each_layer: bool = True,
        pos_enc_type = 'sinusoidal', # 'sinusoidal' or 'learned' or 'RoPE
        tie_weights: bool = True,
        loss_ignore_idx: int = -1):
        """Seq2Seq Encoder-Decoder Dual Attention Transformer

        Parameters
        ----------
        input_spec : dict
            description of input format. dictionary with key 'type' with values 'token' or 'vector'.
            if 'token', must also have 'vocab_size'. if 'vector', must also have 'dim'
        output_spec : dict
            description of output format. dictionary with key 'type' with values 'token' or 'vector'.
            if 'token', must also have 'vocab_size'. if 'vector', must also have 'dim'
        symbol_retrieval : str
            type of symbol retrieval mechanism. must be one of 'symbolic_attention', 'rel_sym_attn', 'positional_symbols', or 'position_relative'
        symbol_retrieval_kwargs : dict
            keyword arguments for symbol retrieval mechanism
        d_model : int
            model dimension
        out_dim : int
            output dimension (e.g., output vocab size)
        n_layers_enc : int
            number of encoder layers
        n_layers_dec : int
            number of decoder layers
        encoder_kwargs : dict
            keyword arguments for encoder blocks
        decoder_kwargs : dict
            keyword arguments for decoder blocks
        in_block_size : int
            block size for input sequence
        out_block_size : int
            block size for target sequence
        pos_enc_type : str, optional
            type of positional encoding to use. must be one of 'sinusoidal', 'learned', or 'RoPE, by default 'sinusoidal'.
        tie_weights : bool, optional
            whether to tie weights between target embedder and final layer weights, by default True
        loss_ignore_idx : int, optional
            idx of class to ignore when computing loss, by default -1

        """
        super().__init__()

        self.input_spec = input_spec
        self.output_spec = output_spec
        self.d_model = d_model
        self.out_dim = out_dim
        self.n_layers_enc = n_layers_enc
        self.n_layers_dec = n_layers_dec
        self.encoder_kwargs = encoder_kwargs
        self.decoder_kwargs = decoder_kwargs
        self.in_block_size = in_block_size
        self.out_block_size = out_block_size
        self.pos_enc_type = pos_enc_type
        self.loss_ignore_idx = loss_ignore_idx
        self.update_symbols_each_layer = update_symbols_each_layer
        self.n_heads = encoder_kwargs['n_heads_sa'] + encoder_kwargs['n_heads_ra'] # assume same number of heads for encoder and decoder
        self.d_head = d_model // self.n_heads

        if input_spec['type'] == 'token':
            source_embedder = torch.nn.Embedding(input_spec['vocab_size'], d_model)
        elif input_spec['type'] == 'vector':
            source_embedder = torch.nn.Linear(input_spec['dim'], d_model)
        else:
            raise ValueError(f"input_spec['type'] must be 'token' or 'vector', not {input_spec['type']}")

        # TODO: add option to share embedder between source and target' maybe via output_spec?
        if output_spec['type'] == 'token':
            target_embedder = torch.nn.Embedding(output_spec['vocab_size'], d_model)
        elif output_spec['type'] == 'vector':
            target_embedder = torch.nn.Linear(output_spec['dim'], d_model)
        else:
            raise ValueError(f"output_spec['type'] must be 'token' or 'vector', not {output_spec['type']}")

        if symbol_retrieval == 'symbolic_attention':
            symbol_retriever = SymbolicAttention(**symbol_retrieval_kwargs)
        elif symbol_retrieval == 'rel_sym_attn':
            symbol_retriever = RelationalSymbolicAttention(**symbol_retrieval_kwargs)
        elif symbol_retrieval == 'positional_symbols':
            symbol_retriever = PositionalSymbolRetriever(**symbol_retrieval_kwargs)
        elif symbol_retrieval == 'position_relative':
            symbol_retriever = PositionRelativeSymbolRetriever(**symbol_retrieval_kwargs)
        else:
            raise ValueError(f"`symbol_retrieval` must be one of 'symbolic_attention', 'rel_sym_attn', or 'positional_symbols'. received {symbol_retrieval}")

        layer_dict = dict(
            source_embedder = source_embedder,
            target_embedder = target_embedder,
            symbol_retriever = symbol_retriever,
            encoder_blocks = nn.ModuleList([DualAttnEncoderBlock(d_model, **encoder_kwargs) for _ in range(n_layers_enc)]),
            decoder_blocks = nn.ModuleList([DualAttnDecoderBlock(d_model, **decoder_kwargs) for _ in range(n_layers_enc)]),
            final_out = nn.Linear(d_model, out_dim)
        )

        if pos_enc_type == 'sinusoidal':
            layer_dict['source_pos_embedder'] = SinusoidalPositionalEncoding(d_model, dropout=0., max_len=in_block_size)
            layer_dict['target_pos_embedder'] = SinusoidalPositionalEncoding(d_model, dropout=0., max_len=out_block_size)
        elif pos_enc_type == 'learned':
            layer_dict['source_pos_embedder'] = LearnedPositionalEmbeddings(d_model, max_len=in_block_size)
            layer_dict['target_pos_embedder'] = LearnedPositionalEmbeddings(d_model, max_len=out_block_size)
        elif pos_enc_type == 'RoPE':
            # if using RoPE, precompute RoPE sine-cosine rotation matrices
            freqs_cos, freqs_sin = precompute_freqs_cis(self.d_head, max(self.in_block_size, self.out_block_size))
            self.register_buffer("freqs_cos", freqs_cos, persistent=False)
            self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        else:
            raise ValueError('`pos_enc_type` invalid')

        self.layers = nn.ModuleDict(layer_dict)

        # weight-tying embedder and final layer
        if tie_weights:
            self.layers.target_embedder.weights = self.layers.final_out


    def forward(self, x, y, targets=None):

        x = self.layers.source_embedder(x)
        y = self.layers.target_embedder(y)

        if self.pos_enc_type == 'sinusoidal' or self.pos_enc_type == 'learned':
            # if using positional embeddings, add positional embeeddings
            x = self.layers.source_pos_embedder(x)
            y = self.layers.target_pos_embedder(y)
            freqs_cos_x, freqs_sin_x, freqs_cos_y, freqs_sin_y = None, None, None, None # not using RoPE
        elif self.pos_enc_type == 'RoPE':
            # otherwise, get the RoPE matrices
            tx, ty = x.size(1), y.size(1)
            freqs_cos_x = self.freqs_cos[:tx]
            freqs_sin_x = self.freqs_sin[:tx]
            freqs_cos_y = self.freqs_cos[:ty]
            freqs_sin_y = self.freqs_sin[:ty]

        symbols = self.layers.symbol_retriever(x)
        for enc_block in self.layers.encoder_blocks:
            if self.update_symbols_each_layer:
                symbols = self.layers.symbol_retriever(x)
            x = enc_block(x, symbols, freqs_cos=freqs_cos_x, freqs_sin=freqs_sin_x)

        symbols = self.layers.symbol_retriever(y)
        for dec_block in self.layers.decoder_blocks:
            if self.update_symbols_each_layer:
                symbols = self.layers.symbol_retriever(y)
            y = dec_block(y, x, symbols=symbols, freqs_cos=freqs_cos_y, freqs_sin=freqs_sin_y)

        if targets is not None:
            # compute loss if given targets
            logits = self.layers.final_out(y)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.loss_ignore_idx)
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
        # NOTE: Model Flops Utilization (MFU) is a measure of how much of the peak FLOPS of the GPU is being utilized.
        # PaLM paper has computed this for standard Transformers
        # haven't done this yet for encoder-decoder architectures, so this is a placeholder

        return -1.0


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
