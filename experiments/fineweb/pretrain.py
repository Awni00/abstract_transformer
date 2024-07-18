# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

import os
import math
import time
from datetime import datetime
import wandb
import argparse
import torch
import torchinfo
from contextlib import nullcontext
from hellaswag import render_example, iterate_examples, get_most_likely_row
from fineweb.fineweb_dataloader import DataLoaderLite
import tiktoken
import gc

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# from language_models import TransformerLM,  configure_optimizers
# from gpt2_model import GPT, GPTConfig
from language_models import TransformerLM, configure_optimizers
from dual_attention_transformer import DualAttnTransformerLM

# Create argument parser
parser = argparse.ArgumentParser(description='Pretrain script for transformer_residual_stream')

# Add arguments for logging and checkpointing
parser.add_argument('--wandb_log', type=int, default=1, help='Enable wandb logging')
parser.add_argument('--wandb_watch', type=int, default=0, help='Enable wandb.watch for logging grads, weights, etc.')
parser.add_argument('--wandb_project', type=str, default='fineweb', help='Wandb project name')
parser.add_argument('--run_name', type=str, default=None, help='Name of the run')

# Add arguments for optimizer configuration
parser.add_argument('--total_batch_size', type=int, default=524_288, help='Total batch size')
parser.add_argument('--B', type=int, default=32, help='Micro batch size')
parser.add_argument('--T', type=int, default=1024, help='Sequence length')
parser.add_argument('--max_lr', type=float, default=6e-4, help='Maximum learning rate')
parser.add_argument('--min_lr', type=float, default=6e-5, help='Minimum learning rate')
parser.add_argument('--warmup_steps', type=int, default=715, help='Number of warmup steps')
parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.95), help='Betas for Adam optimizer')
parser.add_argument('--learning_rate', type=float, default=6e-4, help='Learning rate')
parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')

# Add arguments about training procedure
parser.add_argument('--train_log_interval', type=int, default=10, help='Interval for evaluation')
parser.add_argument('--eval_interval', type=int, default=250, help='Interval for evaluation')
parser.add_argument('--val_loss_steps', type=int, default=20, help='Number of steps for validation loss')
parser.add_argument('--save_checkpoints', type=int, default=1, help='Whether to save checkpoints')
parser.add_argument('--save_interval', type=int, default=5000, help='Interval for saving model')
parser.add_argument('--generate_during_training', type=int, default=0, help='Whether to periodically generate samples during training')
parser.add_argument('--gen_interval', type=int, default=2500, help='Interval for generation')
parser.add_argument('--hellaswag_during_training', type=int, default=0, help='Whether to evaluate hellaswag benchmark')
parser.add_argument('--hellaswag_interval', type=int, default=2500, help='Whether to evaluate hellaswag benchmark')
parser.add_argument('--track_grad_norms', type=int, default=1, help='Whether to track grad norms during trainign')
parser.add_argument('--max_steps', type=int, default=19073, help='Maximum number of steps') # TODO change to calculate dynamically as # of tokens?

# Add arguments for cuda optimizations
parser.add_argument('--compile', type=int, default=1, help='Enable torch.compile')
parser.add_argument('--use_tf32_matmul', type=int, default=1, help='Use bfloat16 for matmuls')
parser.add_argument('--use_bf16', type=int, default=1, help='Use bfloat16 for matmuls')

# Add arguments for model configuration
# default config matches GPT2-medium / GPT3-medium (124M params)
parser.add_argument('--vocab_size', type=int, default=50304, help='Number of tokens in the tokenizer')
parser.add_argument('--d_model', type=int, default=768, help='Dimensionality of the model')
parser.add_argument('--n_layers', type=int, default=12, help='Number of layers in the model')
parser.add_argument('--sa', type=int, default=6, help='Number of attention heads')
parser.add_argument('--ra', type=int, default=6, help='Number of attention heads')
parser.add_argument('--n_kv_heads', type=int, default=None, help='Number of key/value heads (e.g., MQA if 1)')
parser.add_argument('--n_relations', type=int, default=None, help='Number of relations')
parser.add_argument('--rel_activation', type=str, default='identity', help='Relation activation function')
parser.add_argument('--symbol_type', default='symbolic_attention', type=str, choices=('position_relative', 'symbolic_attention', 'NA'), help='type of symbols to use')
parser.add_argument('--trainable_symbols', default=0, type=int, help='whether to make symbols trainable (only applies to symbolic_attention)')
parser.add_argument('--shared_symbol_retriever', default=1, type=int, help='Whether to use a shared symbol retriever for all layers')
parser.add_argument('--weight_tie_symbol_library', default=0, type=int, help='whether to tie weights of symbol library if retriever not shared')
parser.add_argument('--sym_attn_n_symbols', default=None, type=int, help='number of symbols to use in sym_attn')
parser.add_argument('--sym_attn_n_heads', default=None, type=int, help='number of heads to use in sym_attn')
parser.add_argument('--symmetric_rels', default=0, type=int, help='whether to impose symmetric relations in RA')
parser.add_argument('--dff', type=int, default=None, help='Dimensionality of the feed-forward layer')
parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--norm_first', type=int, default=1, help='Whether to apply layer normalization before the attention layer')
parser.add_argument('--norm_type', type=str, default='layernorm', help='Type of normalization')
parser.add_argument('--max_block_size', type=int, default=1024, help='Maximum block size')
parser.add_argument('--bias', type=int, default=0, help='Whether to include bias in the model')
parser.add_argument('--pos_enc_type', type=str, default='RoPE', help='Type of positional encoding')

parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
parser.add_argument('--wandb_fork_run_id', type=str, default=None, help='wandb run id to fork from')

parser.add_argument('--seed', type=int, default=None, help='Random seed')


# -----------------------------------------------------------------------------
# Parse arguments
args = parser.parse_args()

# set seed, if specified
if args.seed is not None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

# TODO: add train config set up? e.g., read from config file?

# Logging and checkpointing configuration
wandb_log = bool(args.wandb_log)
wandb_watch = bool(args.wandb_watch)
wandb_project = args.wandb_project
run_name = args.run_name

# Optimizer configuration
total_batch_size = args.total_batch_size
micro_batch_size = args.B # micro batch size
max_seq_len = args.T # sequence length
max_lr = args.max_lr
min_lr = args.min_lr
warmup_steps = args.warmup_steps
weight_decay = args.weight_decay
betas = tuple(args.betas)
learning_rate = args.learning_rate # TODO: what is difference bw max_lr and learning_rate?
grad_clip = args.grad_clip if args.grad_clip > 0 else None
optimizer_config = dict(
    total_batch_size=total_batch_size, micro_batch_size=micro_batch_size, max_seq_len=max_seq_len, max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps,
    weight_decay=weight_decay, betas=betas, learning_rate=learning_rate, grad_clip=grad_clip)

# Training procedure
resume = args.resume
wandb_fork_run_id = args.wandb_fork_run_id
train_log_interval = args.train_log_interval
eval_interval = args.eval_interval
save_interval = args.save_interval
val_loss_steps = args.val_loss_steps
gen_interval = args.gen_interval
hellaswag_during_training = bool(args.hellaswag_during_training)
hellaswag_interval = args.hellaswag_interval
generate_during_training = bool(args.generate_during_training)
track_grad_norms = bool(args.track_grad_norms)
max_steps = args.max_steps

# CUDA optimizations
use_compile = bool(args.compile)
use_bf16 = bool(args.use_bf16)
use_tf32_matmul = bool(args.use_tf32_matmul)

training_config = dict(max_steps=max_steps, eval_interval=eval_interval, save_interval=save_interval, val_loss_steps=val_loss_steps,
    generate_during_training=generate_during_training, gen_interval=gen_interval,
    hellaswag_during_training=hellaswag_during_training, hellaswag_interval=hellaswag_interval,
    track_grad_norms=track_grad_norms, use_compile=use_compile, use_bf16=use_bf16, use_tf32_matmul=use_tf32_matmul)

# Model configuration
vocab_size = args.vocab_size
d_model = args.d_model
n_layers = args.n_layers
sa, ra = args.sa, args.ra
dff = args.dff
ra_type = 'relational_attention'
symmetric_rels = bool(args.symmetric_rels) if args.symmetric_rels in (0,1) else None
n_relations = args.n_relations
rel_proj_dim = None if n_relations is None else int((d_model / (sa+ra)) * (ra / n_relations))
rel_activation = args.rel_activation
symbol_type = args.symbol_type
trainable_symbols = bool(args.trainable_symbols)
sym_attn_n_symbols = args.sym_attn_n_symbols if args.sym_attn_n_symbols is not None else d_model # only applicable for symbol_type=sym_attn
sym_attn_n_heads = args.sym_attn_n_heads if args.sym_attn_n_heads is not None else 4 # only applicable for symbol_type=sym_attn
symbol_retriever_config = dict(shared_symbol_retriever=bool(args.shared_symbol_retriever), weight_tie_symbol_library=bool(args.weight_tie_symbol_library))
activation = args.activation
dropout_rate = args.dropout_rate
norm_first = bool(args.norm_first)
norm_type = args.norm_type
max_block_size = args.max_block_size
bias = bool(args.bias)
pos_enc_type = args.pos_enc_type

ra_kwargs = dict(n_relations=n_relations, rel_activation=rel_activation, rel_proj_dim=rel_proj_dim, n_kv_heads=args.n_kv_heads)
sa_kwargs = dict(n_kv_heads=args.n_kv_heads)
if symbol_type == 'symbolic_attention':
    # NOTE: n_heads, n_symbols fixed for now
    symbol_retrieval_kwargs = dict(d_model=d_model, n_symbols=sym_attn_n_symbols, n_heads=sym_attn_n_heads, trainable_symbols=trainable_symbols)
elif symbol_type == 'position_relative':
    symbol_retrieval_kwargs = dict(symbol_dim=d_model, max_rel_pos=max_seq_len)
    ra_kwargs['use_relative_positional_symbols'] = True # if using position-relative symbols, need to tell RA module
elif ra != 0:
    raise ValueError(f'`symbol_type` {symbol_type} not valid')

if ra_type == 'relational_attention':
    ra_kwargs['symmetric_rels'] = symmetric_rels

# if ra=0, use TransformerLM
if ra == 0:
    model_config = dict(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=sa, dff=dff,
        pos_enc_type=pos_enc_type, dropout_rate=dropout_rate, activation=activation, norm_first=norm_first,
        max_block_size=max_seq_len, bias=bias, use_flash_attention=True)

    # model = TransformerLM(**model_args).to(device)
# otherwise, use DualAttnTransformerLM
else:
    model_config = dict(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads_sa=sa, n_heads_ra=ra, dff=dff,
        sa_kwargs=sa_kwargs, ra_kwargs=ra_kwargs, ra_type=ra_type,
        symbol_retrieval=symbol_type, symbol_retrieval_kwargs=symbol_retrieval_kwargs, symbol_retriever_config=symbol_retriever_config,
        pos_enc_type=pos_enc_type, activation=activation,
        dropout_rate=dropout_rate, norm_first=norm_first, max_block_size=max_seq_len, bias=bias)

    # model = DualAttnTransformerLM(**model_args).to(device)

run_config = dict(
    optimizer_config=optimizer_config, training_config=training_config, model_config=model_config)

# annotate run_name with datetime
datetime_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
run_name = datetime_now if run_name is None else f'{run_name}_{datetime_now}'

# -----------------------------------------------------------------------------
# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0) # master process will do logging, checkpointing etc.
else:
    # non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

if master_process and torch.cuda.is_available():
    print("CUDA is available")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print("GPU Types:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    if ddp:
        print("Running in DDP mode")
else:
    print("CUDA is not available")


# tokenizer
enc = tiktoken.get_encoding("gpt2")

# calculate gradient accumulation steps
assert total_batch_size % (micro_batch_size * max_seq_len * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (micro_batch_size * max_seq_len * ddp_world_size)
run_config['training_config']['grad_accum_steps'] = grad_accum_steps

if master_process:
    print(f"total batch size: {total_batch_size} tokens")
    print(f'sequence length: {max_seq_len} tokens')
    print(f'# of processes: {ddp_world_size}')
    print(f'micro batch size: {micro_batch_size} batches')
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=micro_batch_size, T=max_seq_len, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
if master_process:
    print(f"found {len(train_loader.shards)} shards for trainsplit")

val_loader = DataLoaderLite(B=micro_batch_size, T=max_seq_len, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")
if master_process:
    print(f"found {len(val_loader.shards)} shards for trainsplit")

# set up torch to use bfloat16 for matmuls
if use_tf32_matmul:
    torch.set_float32_matmul_precision('high')
autocast_ctx_manager = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if use_bf16 else nullcontext()

# create model
start_step = 0
if resume is not None:
    print('loading checkpoint and extracting model_config')
    ckpt = torch.load(resume, map_location=device)
    start_step = ckpt['step']
    model_config = ckpt['config']
    run_config = ckpt.get('run_config', None)

print('building model')
if 'n_heads_ra' in model_config:
    model = DualAttnTransformerLM(**model_config)
else:
    model = TransformerLM(**model_config)

if resume is not None:
    print('loading model weights from checkpoint')
    state_dict = ckpt['model']
    prefix_to_remove = '_orig_mod.'
    model_state_dict = {k.replace(prefix_to_remove, ''): v for k, v in state_dict.items()}
    model.load_state_dict(model_state_dict)

model = model.to(device)
model_summary = torchinfo.summary(model, input_data=torch.zeros((1, max_seq_len), device=device).int())

model_summary_dict = {
    'Input size (MB)': model_summary.to_megabytes(model_summary.total_input),
    'Params size (MB)': model_summary.to_megabytes(model_summary.total_param_bytes),
    'Forward/backward pass size  (MB)': model_summary.to_megabytes(model_summary.total_output_bytes),
    'Estimated total size (MB)': model_summary.to_megabytes(model_summary.total_output_bytes + model_summary.total_param_bytes + model_summary.total_input),
    'Total Mult-Adds': model_summary.total_mult_adds,

    'trainable_params': model_summary.trainable_params, # note: numbers from torchinfo are not always accurate
    'total_params': model_summary.total_params, # note: numbers from torchinfo are not always accurate

    'num_params': sum(p.numel() for p in model.parameters()),
    'num_trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
}
if master_process:
    print(model_summary)
    print(f'num params: {model_summary_dict["num_params"]:,}')
    print(f'num trainable params: {model_summary_dict["num_trainable_params"]:,}')

run_config['model_summary'] = model_summary_dict

# torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

def get_lr(it):
    # stage 1: linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # stage 3: if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # stage 2: in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# initialize wandb logging
if wandb_log and master_process:
    if wandb_fork_run_id is None:
        wandb.init(project=wandb_project, name=run_name, config=run_config)
    else:
        print('Forking from previous W&B run with ID:', wandb_fork_run_id)
        wandb.init(project=wandb_project, fork_from=f'{wandb_fork_run_id}?_step={start_step}')

# optimizer
optimizer = configure_optimizers(raw_model, weight_decay=weight_decay, betas=betas,
    learning_rate=learning_rate, device_type=device_type)

# load optimizer state from checkpoint if resuming
if resume is not None:
    optimizer.load_state_dict(ckpt['optimizer'])

# create the log directory we will write checkpoints to and log to
log_dir = f"log/{run_name}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log_{run_name}.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

# region training loop utility functions

# TODO: modify to take these things as input?
def eval_val_loss():
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with autocast_ctx_manager:
                logits, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()

    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

    return val_loss_accum

def save_checkpoint():
    # optionally write model checkpoints
    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
    checkpoint = {
        'model': raw_model.state_dict(),
        'config': model_config,
        'run_config': run_config,
        'step': step,
        'val_loss': val_loss_accum.item(),
        'optimizer': optimizer.state_dict(),
        'seed': args.seed
    }
    torch.save(checkpoint, checkpoint_path)

def eval_hellaswag():
    num_correct_norm = 0
    num_total = 0

    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.no_grad():
            with autocast_ctx_manager:
                # NOTE: using compiled model seems to work on a single-GPU with an A40
                # but it gives a torchdynamo error with DDP and an A100/H100 (not sure why)
                # logits, loss = model(tokens)
                logits, loss = raw_model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    del tokens, mask, label, logits

    # reduce the stats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
        print(f"step {step:5d} | HellaSwag acc {num_correct_norm}/{num_total}={acc_norm:.4f}")
        with open(log_file, "a") as f:
            f.write(f"step {step:5d} | HellaSwag acc {num_correct_norm}/{num_total}={acc_norm:.4f}\n")

        if wandb_log:
            try:
                wandb.log({"hellaswag/acc": acc_norm}, step = step)
            except Exception as e:
                print(f"logging to wandb failed: {e}")

def generate_samples():
    model.eval()
    num_return_sequences = 4 # TODO: make these configurable parameters
    max_length = 32
    tokens = enc.encode("Hello, I'm a language model,") # TODO: make this more interesting
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)

    while xgen.size(1) < max_length:
        with torch.no_grad():
            with autocast_ctx_manager:
                logits, loss = model(xgen) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            xgen = torch.cat((xgen, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"rank {ddp_rank} sample {i}: {decoded}")

    del xgen, logits, loss, probs, topk_probs, topk_indices, ix, xcol

# utility function for getting the gradient norms

def get_layerwise_grad_norms(model):
    # get grad norm for each layer / block (viewed as a single vector)
    layerwise_grad_norms = dict()
    for l, block in enumerate(model.layers.blocks):
        layer_l_grad_norm = torch.concat([p.grad.flatten() for p in block.parameters() if p.grad is not None]).norm(p=2)
        layerwise_grad_norms[f'block_{l+1}'] = layer_l_grad_norm.detach().item()
    return layerwise_grad_norms

def get_grad_norms(model):
    # get grad norm for each parameter
    return {pn: torch.norm(p.grad, p='fro').detach().item() for pn, p in model.layers.named_parameters() if p.grad is not None}

# utility function for formatting time
def format_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.2f} ms "
    elif seconds < 60:
        return f"{seconds:.2f} sec"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hr "

# endregion

# NOTE: W&B's implementation logs histogram of gradients for all parameters; probably too much for large models?
# the track_grad_norms option is a simpler alternative I implemented that only logs the norm of the gradients per parameter
if wandb_log and wandb_watch:
    wandb.watch(raw_model, log="all", log_freq=eval_interval*grad_accum_steps, log_graph=True) # log gradients, weights, biases, etc.

# resume training if checkpoint is provided
# step counter starts from last step and the train_loader is set to the correct position
# NOTE: this assumes that total_batch_size is the same as was used in previous run (if not in ckpt run_config)
if resume is not None:
    if start_step > max_steps:
        print(f"checkpoint step {start_step} is greater than max_steps {max_steps}, exiting")
        exit()
    if 'run_config' in ckpt:
        total_batch_size_ = ckpt['run_config']['optimizer_config']['total_batch_size']
        token_position = start_step * total_batch_size_
        train_loader.set_current_position(token_position)
        del total_batch_size_
    else:
        print("WARNING: could not find total_batch_size in checkpoint, assuming previous run used the same total_batch_size")
        token_position = start_step * total_batch_size
        train_loader.set_current_position(token_position)

    print(f'RESUMING TRAINING FROM STEP {start_step}')
    print(f'START POSITION IN `train_loader` is {train_loader.current_position:,}')
    print(f"NOTE: THIS WAS CALCULATED BASED ON ckpt['step'] AND `total_batch_size`")
    print()
else:
    start_step = 1

start_time = time.time()
for step in range(start_step, max_steps + 1):
    t0 = time.time()
    last_step = (step == max_steps)

    # once in a while evaluate our validation loss
    if step % eval_interval == 0 or last_step:
        val_loss_accum = eval_val_loss()

        if master_process:
            # compute and log the gradient norms (currently only for the master process)

            print(f"step {step:5d} | val loss {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"step {step:5d} | val loss {val_loss_accum.item():.4f}\n")

            if wandb_log:
                try:
                    wandb.log({"loss/val": val_loss_accum.item()}, step = step)

                    if step > 0 and track_grad_norms:
                        param_grad_norms = get_grad_norms(raw_model)
                        layer_grad_norms = get_layerwise_grad_norms(raw_model)
                        wandb.log({f'grad_norms/{pn}': gn for pn, gn in param_grad_norms.items()}, step = step)
                        wandb.log({f'layer_grad_norms/{ln}': gn for ln, gn in layer_grad_norms.items()}, step = step)
                        del param_grad_norms, layer_grad_norms
                except Exception as e:
                    print(f"logging to wandb failed: {e}")

            if step > 0 and (step % save_interval == 0 or last_step) and args.save_checkpoints:
                save_checkpoint()

    # once in a while evaluate hellaswag
    if (step % hellaswag_interval == 0 or last_step) and hellaswag_during_training:
        eval_hellaswag()

    # # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % gen_interval == 0) or last_step) and generate_during_training:
        generate_samples()

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # set up DDP syncing of accumulated gradients after final microbatch
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with autocast_ctx_manager:
            logits, loss = model(x, y)
        # note that the loss is scaled by the gradient accum steps 
        # because loss.backward() adds accumulated loss grads (not mean, which is what we want)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # clip gradients
    if grad_clip is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # TODO: log norms per layer

    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # take an optimization steps according to loss gradient
    optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work

    # log the training stats
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    elapsed_time = t1 - start_time # elapsed time in minutes
    percent_progress = step / max_steps
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    eta = (max_steps - step) * dt # estimated time until completion
    if master_process:
        log_string = f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {grad_norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | progress: {percent_progress*100:.2f}% | elapsed: {format_time(elapsed_time)} | ETA: {format_time(eta)}"
        print(log_string)
        with open(log_file, "a") as f:
            f.write(log_string + "\n")

        # log to wandb
        if wandb_log and step % train_log_interval == 0:
            try:
                wandb.log(
                    {
                        "step": step,
                        "tokens": step * grad_accum_steps * ddp_world_size * micro_batch_size * max_seq_len,
                        "loss/train": loss_accum.item(),
                        "grad_norm": grad_norm,
                        "lr": lr,
                        "tokens_per_sec": tokens_per_sec,
                        # "mfu": running_mfu * 100,  # convert to percentage
                    }, step = step
                )
            except Exception as e:
                print(f"logging to wandb failed: {e}")
    del loss_accum, loss, logits, x, y
    gc.collect()

if wandb_log and master_process:
    wandb.finish()

if ddp:
    destroy_process_group()

# TODO: incorporate missing features in llama2.c's training loop into this one