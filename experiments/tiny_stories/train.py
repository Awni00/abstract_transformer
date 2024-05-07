"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU small debug run, example:
$ python -m train.py --compile=False --eval_iters=10 --batch_size=8

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
import argparse
from tokenizer import Tokenizer
import numpy as np

# os.environ['TORCH_LOGS'] = "+dynamo"
# os.environ['TORCHDYNAMO_VERBOSE'] = "1"



import torch
import torchmetrics
import torchinfo

from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# from model import Transformer, ModelArgs
import sys; sys.path.append('../..')
from language_models import TransformerLM, AbstractTransformerLM, configure_optimizers

from tiny_stories_data import Task
# from export import model_export
parser = argparse.ArgumentParser()

parser.add_argument('--sa', required=True, type=int, help='number of self-attention heads')
parser.add_argument('--rca', required=True, type=int, help='number of relational cross-attention heads')
parser.add_argument('--symbol_type', required=True, type=str, choices=('pos_relative', 'sym_attn', 'NA'), help='type of symbols to use')
parser.add_argument('--pos_enc_type', required=True, type=str, choices=('RoPE', 'pos_emb'), help='type of symbols to use')
parser.add_argument('--rca_type', required=True, type=str, choices=('standard', 'disentangled_v1', 'disentangled_v2', 'NA'), help="type of RCA to use")
parser.add_argument('--n_layers', required=True, type=int, help='number of layers')
parser.add_argument('--d_model', required=True, type=int, help='model dimension')
parser.add_argument('--activation', default='swiglu', type=str, help='MLP activation')
parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')
parser.add_argument('--dff', default=None, type=int, help='feedforward hidden dimension')
parser.add_argument('--max_seq_len', default=512, type=int, help='max seq length / block size')

parser.add_argument('--n_epochs', default=-1, type=int, help='number of passes through data to train for')
parser.add_argument('--max_iters', default=100_000, type=int, help='maximum number of steps')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='number of gradiient accumulation steps')
parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate')

parser.add_argument('--eval_interval', default=2_000, type=int, help='interval of evaluating validation set')
parser.add_argument('--eval_iters', default=100, type=int, help='# of iters to estimate val loss')
parser.add_argument('--eval_only', default=0, type=int, help='whether to exit after first eval')
parser.add_argument('--log_to_wandb', default=1, type=int, help='whether to log to wandb')
parser.add_argument('--always_save_checkpoint', default=0, type=int, help='whether to save ckpt after each  eval')
parser.add_argument('--compile', default=0, type=int, help='whether to compile')

# parser.add_argument('--run_name', default=None, type=str, help='wandb run name')
parser.add_argument('--wandb_project', default='abstract_transformer--tiny_stories-LM-dev',
    type=str, help='W&B project name')

args = parser.parse_args()

# -----------------------------------------------------------------------------
# I/O
datetime_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

eval_interval = args.eval_interval # 2000
log_interval = 1
eval_iters = args.eval_iters # 100
eval_only = bool(args.eval_only) # False  # if True, script exits right after the first eval
always_save_checkpoint = bool(args.always_save_checkpoint) # False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume'

# wandb logging
wandb_log = bool(args.log_to_wandb) # True  # disabled by default
wandb_project = args.wandb_project # "llamac"

# data
batch_size = args.batch_size # 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = args.max_seq_len # 256
vocab_source = "llama2" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size = 32000 # the Llama 2 tokenizer has 32K tokens

# model
d_model = args.d_model # 64 # 288
n_layers = args.n_layers # 5# 6
sa, rca = args.sa, args.rca
dff = None
rca_type = args.rca_type
symbol_type = args.symbol_type
# n_heads = 8 # 6
# n_kv_heads = 4 # 6
# multiple_of = 32
dropout_rate = args.dropout_rate # 0.0
pos_enc_type = args.pos_enc_type
activation = args.activation
norm_first = True
bias = False
# TODO: add support for norm_type='rmsnorm'

# tokenizer
tokenizer = Tokenizer()


# names of things
if rca > 0:
    model_name = f'sa={sa}; rca={rca}; d={d_model}; L={n_layers}; rca_type={rca_type}; symbol_type={symbol_type}; pos_enc_type={pos_enc_type}'
else:
    model_name = f'sa={sa}; d={d_model}; L={n_layers}; pos_enc_type={pos_enc_type}'

out_dir = f"out/{model_name}__{datetime_now}"
wandb_run_name = f"{model_name}__{datetime_now}"


# adamw optimizer
gradient_accumulation_steps = args.gradient_accumulation_steps # 4  # used to simulate larger batch sizes
learning_rate = args.learning_rate # 5e-4  # max learning rate
max_iters = args.max_iters # 100_000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 1000  # how many steps to warm up for
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "bfloat16"  # float32|bfloat16|float16
compile = bool(args.compile) # False # True FIXME (somehow my models can't be compiled :())  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
# exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# fixing some hyperparams to sensible defaults
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# validating checks
assert vocab_source in ["llama2", "custom"]
assert vocab_source == "custom" or vocab_size == 32000, "The vocab from Meta has 32K tokens"

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# task-specific setup
iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size,
    vocab_source=vocab_source,
    device=device,
    num_workers=0,
)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
# model_args = dict(
#     dim=d_model,
#     n_layers=n_layers,
#     n_heads=n_heads,
#     n_kv_heads=n_kv_heads,
#     vocab_size=vocab_size,
#     multiple_of=multiple_of,
#     max_seq_len=max_seq_len,
#     dropout=dropout,
# )  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # model_args = dict(
    #     vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=sa, dff=dff, pos_enc_type=pos_enc_type,
    #     dropout_rate=dropout, activation=activation, norm_first=norm_first, max_block_size=max_seq_len, bias=bias)
    # model = TransformerLM(**model_args)
    rca_kwargs = dict()
    if symbol_type == 'sym_attn':
        symbol_retrieval_kwargs = dict(d_model=d_model, n_symbols=50, n_heads=4) # NOTE: n_heads, n_symbols fixed for now
    elif symbol_type == 'pos_relative':
        symbol_retrieval_kwargs = dict(symbol_dim=d_model, max_rel_pos=max_seq_len)
        rca_kwargs['use_relative_positional_symbols'] = True # if using position-relative symbols, need to tell RCA module
    elif rca != 0:
        raise ValueError(f'`symbol_type` {symbol_type} not valid')

    # if rca=0, use TransformerLM
    if rca == 0:
        model_args = dict(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=sa, dff=dff,
            pos_enc_type=pos_enc_type, dropout_rate=dropout_rate, activation=activation, norm_first=norm_first,
            max_block_size=max_seq_len, bias=bias)

        model = transformer_lm = TransformerLM(**model_args).to(device)
    # otherwise, use AbstractTransformerLM
    else:
        model_args = dict(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads_sa=sa, n_heads_rca=rca, dff=None,
            rca_kwargs=rca_kwargs, rca_type=rca_type, symbol_retrieval=symbol_type, symbol_retrieval_kwargs=symbol_retrieval_kwargs,
            pos_enc_type=pos_enc_type, activation=activation,
            dropout_rate=dropout_rate, norm_first=norm_first, max_block_size=max_seq_len, bias=bias)

        model = abstracttransformer_lm = AbstractTransformerLM(**model_args).to(device)

    print(torchinfo.summary(model, device='cuda'))
    n_params = sum(p.numel() for p in model.parameters())
    n_params_wo_embedding = n_params - sum(p.numel() for p in model.layers.token_embedder.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total # of params: {n_params}')
    print(f'# of params w/o embedding: {n_params_wo_embedding}')
    config['n_params'] = n_params
    config['n_params_wo_embedding'] = n_params_wo_embedding

    # gptconf = ModelArgs(**model_args)
    # model = Transformer(gptconf)
elif init_from == "resume":
    raise NotImplementedError("haven't implemented this yet")
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
# optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
    # construction time since NCCL does not support `ComplexFloat`
    prefix = "_orig_mod." if compile else ""
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        perplexities = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits, loss = model(X, Y)

                # loss = raw_model.last_loss
                perplexity = torchmetrics.functional.text.perplexity(logits, Y)
            losses[k] = loss.item()
            perplexities[k] = perplexity.item()
        out[f'{split}/loss'] = losses.mean()
        out[f'{split}/perplexity'] = perplexities.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    run = wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train/loss']:.4f}, val loss {losses['val/loss']:.4f}, train perpl {losses['train/perplexity']:.4f}, val perpl {losses['val/perplexity']:.4f}")
        if wandb_log:
            try:
                wandb.log(
                    {
                        "iter": iter_num,
                        "tokens": iter_num * tokens_per_iter,
                        "train/loss": losses["train/loss"],
                        "val/loss": losses["val/loss"],
                        "train/perplexity": losses["train/perplexity"],
                        "val/perplexity": losses["val/perplexity"],
                        "lr": lr,
                        "mfu": running_mfu * 100,  # convert to percentage
                    }, step = iter_num
                )
            except Exception as e:
                print(f"logging to wandb failed: {e}")
        if losses["val/loss"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val/loss"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                # model_export(raw_model, os.path.join(out_dir, "model.bin"), version=0)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            logits, loss = model(X, Y)

            # loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = next(train_batch_iter)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()


# region generate some samples from fitted model then finish
prompts = [
    'Once upon a time,',
    'There once was a girl named ',
    'On a rainy day,',
    'Emma went to',
    '',
    '',
    '',
]

def generate_from_prompt(model, prompt, max_new_tokens=100, temperature=1.0, top_k=None, tokenizer=tokenizer):
    prompt_idx = torch.from_numpy(np.array(tokenizer.encode(prompt, eos=True, bos=True))).unsqueeze(0)#.to(device)
    prompt_idx = prompt_idx[:, :-1] # remove final token because it is [SEP]
    sample_gen = model.generate(prompt_idx.to(device), max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)[0]
    sample_gen = tokenizer.decode(sample_gen.tolist())
    return sample_gen


print()
print('='*100)
print("GENERATING SAMPLES")
samples = []
for prompt in prompts:
    print(f"PROMPT: {prompt}")
    print("GENERATED TEXT:")
    sample_gen = generate_from_prompt(model, prompt)
    print(sample_gen)
    print('-'*100)
    print()
    samples.append(sample_gen)

if wandb_log:
    samples_table = [[p, g] for p, g in zip(prompts, samples)]
    samples_table = wandb.Table(columns=["Prompt", "Generated Sample"], data = samples_table)
    run.log({"Generated Samples": samples_table})

    wandb.finish()
# endregion