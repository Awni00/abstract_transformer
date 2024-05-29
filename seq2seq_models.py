import torch
from torch import nn

from transformer_blocks import EncoderBlock, DecoderBlock
from dual_attn_blocks import DualAttnEncoderBlock, DualAttnDecoderBlock
from symbol_retrieval import SymbolicAttention, RelationalSymbolicAttention, PositionalSymbolRetriever, PositionRelativeSymbolRetriever
from positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEmbeddings

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
        self.loss_ignore_idx = loss_ignore_idx

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
            decoder_blocks = nn.ModuleList([DecoderBlock(d_model, **decoder_kwargs) for _ in range(n_layers_dec)]),
            final_out = nn.Linear(d_model, out_dim)
        )

        self.layers = nn.ModuleDict(layer_dict)

        # weight-tying embedder and final layer
        if tie_weights:
            self.layers.target_embedder.weights = self.layers.final_out



    def forward(self, x, y, targets=None):

        x = self.layers.source_embedder(x)
        y = self.layers.target_embedder(y)
        x = self.layers.source_pos_embedder(x)
        y = self.layers.target_pos_embedder(y)

        for enc_block in self.layers.encoder_blocks:
            x = enc_block(x)

        for dec_block in self.layers.decoder_blocks:
            y = dec_block(y, x)

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
            type of symbol retrieval mechanism. must be one of 'sym_attn', 'rel_sym_attn', 'pos_sym_retriever', or 'pos_relative'
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
        self.loss_ignore_idx = loss_ignore_idx

        # TODO: make positional embedder configurable (learned or fixed sinusoidal, etc)
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

        if symbol_retrieval == 'sym_attn':
            symbol_retriever = SymbolicAttention(**symbol_retrieval_kwargs)
        elif symbol_retrieval == 'rel_sym_attn':
            symbol_retriever = RelationalSymbolicAttention(**symbol_retrieval_kwargs)
        elif symbol_retrieval == 'pos_sym_retriever':
            symbol_retriever = PositionalSymbolRetriever(**symbol_retrieval_kwargs)
        elif symbol_retrieval == 'pos_relative':
            symbol_retriever = PositionRelativeSymbolRetriever(**symbol_retrieval_kwargs)
        else:
            raise ValueError(f"`symbol_retrieval` must be one of 'sym_attn', 'rel_sym_attn', or 'pos_sym_retriever'. received {symbol_retrieval}")


        layer_dict = dict(
            source_embedder = source_embedder,
            target_embedder = target_embedder,
            source_pos_embedder = SinusoidalPositionalEncoding(d_model, dropout=0., max_len=in_block_size),
            target_pos_embedder = SinusoidalPositionalEncoding(d_model, dropout=0., max_len=out_block_size),
            symbol_retriever = symbol_retriever,
            # dropout = nn.Dropout(dropout_rate),
            encoder_blocks = nn.ModuleList([DualAttnEncoderBlock(d_model, **encoder_kwargs) for _ in range(n_layers_enc)]),
            decoder_blocks = nn.ModuleList([DualAttnDecoderBlock(d_model, **decoder_kwargs) for _ in range(n_layers_enc)]),
            final_out = nn.Linear(d_model, out_dim)
        )

        self.layers = nn.ModuleDict(layer_dict)

        # weight-tying embedder and final layer
        if tie_weights:
            self.layers.target_embedder.weights = self.layers.final_out


    def forward(self, x, y, targets=None):

        x = self.layers.source_embedder(x)
        y = self.layers.target_embedder(y)
        x = self.layers.source_pos_embedder(x)
        y = self.layers.target_pos_embedder(y)

        for enc_block in self.layers.encoder_blocks:
            symbols = self.layers.symbol_retriever(x)
            x = enc_block(x, symbols)

        for dec_block in self.layers.decoder_blocks:
            symbols = self.layers.symbol_retriever(y)
            y = dec_block(y, x, symbols=symbols)

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
