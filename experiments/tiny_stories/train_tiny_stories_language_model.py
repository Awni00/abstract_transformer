import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import wandb

import numpy as np
import time
from  tqdm import tqdm

import torch
import torchinfo
import lightning as L
from lightning.pytorch.loggers.wandb import WandbLogger
import torchmetrics
# import tiktoken

import sys; sys.path.append('../..')
from language_models import TransformerLM, AbstractTransformerLM, configure_optimizers
from utils.pl_tqdm_progbar import TQDMProgressBar

print('cuda available: ', torch.cuda.is_available())
print('device count: ', torch.cuda.device_count())
print('current device name: ', torch.cuda.get_device_name(torch.cuda.current_device()))
print('Memory Usage:')
print('\tAllocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
print('\tReserved:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# region parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--sa', required=True, type=int, help='number of self-attention heads')
parser.add_argument('--rca', required=True, type=int, help='number of relational cross-attention heads')
parser.add_argument('--symbol_type', required=True, type=str, choices=('pos_relative', 'sym_attn', 'NA'), help='type of symbols to use')
parser.add_argument('--pos_enc_type', required=True, type=str, choices=('RoPE', 'pos_emb'), help='type of symbols to use')
parser.add_argument('--disentangled_rca', required=True, type=int, help="wehther to use disentangled RCA (0 or 1)")
parser.add_argument('--n_layers', required=True, type=int, help='number of layers')
parser.add_argument('--d_model', required=True, type=int, help='model dimension')
# parser.add_argument('--dff', required=True, type=int, help='feedforward hidden dimension')

parser.add_argument('--n_epochs', default=1, type=int, help='number of passes through data to train for')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
# parser.add_argument('--run_name', default=None, type=str, help='wandb run name')
parser.add_argument('--wandb_project', default='abstract_transformer--tiny_stories-LM',
    type=str, help='W&B project name')

# configuration of PyTorch Lightning Trainer
parser.add_argument('--eval_interval', default=500, type=int, help='interval of evaluating validation set')
parser.add_argument('--log_every_n_steps', default=10, type=int, help='interval of logging training metrics')
parser.add_argument('--max_steps', default=-1, type=int, help='maximum number of steps')
parser.add_argument('--log_model', default=1, type=int, help='whether to save the model at the end of training')
parser.add_argument('--log_to_wandb', default=1, type=int, help='whether to log to wandb')
args = parser.parse_args()

batch_size = args.batch_size
n_epochs = args.n_epochs

# get model config from args (and fix others)
d_model, sa, rca, n_layers = args.d_model, args.sa, args.rca, args.n_layers
dff = None
disentangled_rca = bool(args.disentangled_rca)
symbol_type = args.symbol_type
pos_enc_type = args.pos_enc_type
dropout_rate = 0.2
activation = 'gelu' # gelu rather than relu
norm_first = True
bias = True

run_name = f'sa={sa}; rca={rca}; d={d_model}; L={n_layers}; rca_dis={disentangled_rca}; symbol_type={symbol_type}; pos_enc_type={pos_enc_type}'
# run_name = args.run_name if args.run_name is not None else group_name
group_name = None
wandb_project = args.wandb_project
log_to_wandb = bool(args.log_to_wandb)

eval_interval, max_steps, log_model, log_every_n_steps = args.eval_interval, args.max_steps, bool(args.log_model), args.log_every_n_steps
log_on_step = True # log metrics of training steps (at eval_interval)
# endregion

# region some configuration
device = 'cuda'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# optimization hyperparams
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
decay_lr = True # whether to decay the learning rate
lr_decay_iters = 5000 # make equal to max_iters usually
weight_decay = 1e-1
min_lr = 1e-4 # learning_rate / 10 usually
beta1 = 0.9
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
# warmup_iters = 100
gradient_accumulation_steps = 1 # accumulate gradients over this many steps. simulates larger batch size

# batch size and block size
block_size = 256

# endregion

# region data set up
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("roneneldan/TinyStories")

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

dataset = dataset.map(
    lambda x: tokenizer(x['text'], padding=True, truncation=True, max_length=block_size+1),
    batched=True)

dataset.set_format(type='torch', columns=['input_ids'])

train_dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, pin_memory=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(dataset['validation'], batch_size=batch_size, pin_memory=True, num_workers=4)
# endregion

# region define Pytorch Lightning Model

class LitLanguageModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        text = batch['input_ids']
        x, y = text[:, :-1], text[:, 1:]

        logits, loss = self.model(x, y)
        perplexity = torchmetrics.functional.text.perplexity(logits, y, ignore_index=tokenizer.pad_token_id)

        self.log('train/loss', loss, prog_bar=True, logger=True, on_step=log_on_step, on_epoch=True)
        self.log('train/perplexity', perplexity, prog_bar=True, logger=True, on_step=log_on_step, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        text = batch['input_ids']
        x, y = text[:, :-1], text[:, 1:]

        logits, loss = self.model(x, y)

        perplexity = torchmetrics.functional.text.perplexity(logits, y, ignore_index=tokenizer.pad_token_id)

        self.log(f"val/loss", loss, prog_bar=True, logger=True, add_dataloader_idx=False)
        self.log(f'val/perplexity', perplexity, prog_bar=True, logger=True, add_dataloader_idx=False)

    def test_step(self, batch, batch_idx):
        text = batch['input_ids']
        x, y = text[:, :-1], text[:, 1:]

        logits, loss = self.model(x, y)

        perplexity = torchmetrics.functional.text.perplexity(logits, y, ignore_index=tokenizer.pad_token_id)

        self.log(f"test/loss", loss, prog_bar=True, logger=True, add_dataloader_idx=False)
        self.log(f'test/perplexity', perplexity, prog_bar=True, logger=True, add_dataloader_idx=False)

    def configure_optimizers(self):
        optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type=device)
        return optimizer

# endregion

# define model

# define kwargs for symbol-retrieval module based on type
rca_kwargs = dict()
if symbol_type == 'sym_attn':
    symbol_retrieval_kwargs = dict(d_model=d_model, n_symbols=50, n_heads=4) # NOTE: n_heads, n_symbols fixed for now
elif symbol_type == 'pos_relative':
    symbol_retrieval_kwargs = dict(symbol_dim=d_model, max_rel_pos=block_size)
    rca_kwargs['use_relative_positional_symbols'] = True # if using position-relative symbols, need to tell RCA module
elif rca != 0:
    raise ValueError(f'`symbol_type` {symbol_type} not valid')

# if rca=0, use TransformerLM
if rca == 0:
    model_args = dict(
        vocab_size=tokenizer.vocab_size, d_model=d_model, n_layers=n_layers, n_heads=sa, dff=dff,
        pos_enc_type=pos_enc_type, dropout_rate=dropout_rate, activation=activation, norm_first=norm_first,
        max_block_size=block_size, bias=bias)

    model = transformer_lm = TransformerLM(**model_args).to(device)
# otherwise, use AbstractTransformerLM
else:
    model_args = dict(
        vocab_size=tokenizer.vocab_size, d_model=d_model, n_layers=n_layers, n_heads_sa=sa, n_heads_rca=rca, dff=None,
        rca_kwargs=rca_kwargs, rca_disentangled=disentangled_rca, symbol_retrieval=symbol_type, symbol_retrieval_kwargs=symbol_retrieval_kwargs,
        pos_enc_type=pos_enc_type, activation=activation,
        dropout_rate=dropout_rate, norm_first=norm_first, max_block_size=block_size, bias=bias)

    model = abstracttransformer_lm = AbstractTransformerLM(**model_args).to(device)

print(torchinfo.summary(model, input_data=torch.randint(0, 10, size=(1,block_size)), device='cuda'))

num_params = model.get_num_params()
print(f'# of params {num_params:,}')

# createe LitLanguageModel
lit_model = LitLanguageModel(model)
# endregion

# region train model

if log_to_wandb:
    run = wandb.init(project=wandb_project, group=group_name, name=run_name,
        config={'group': group_name, 'num_params': num_params, **model_args})

    wandb_logger = WandbLogger(experiment=run, log_model=log_model),
else:
    wandb_logger = None

callbacks = [
    TQDMProgressBar(refresh_rate=50)
]

trainer_kwargs = dict(
    max_epochs=n_epochs, enable_checkpointing=False, enable_model_summary=True, benchmark=True,
    enable_progress_bar=True, callbacks=callbacks, logger=wandb_logger,
    accumulate_grad_batches=gradient_accumulation_steps, gradient_clip_val=grad_clip,
    log_every_n_steps=log_every_n_steps, max_steps=max_steps, val_check_interval=eval_interval)

trainer = L.Trainer(
    **trainer_kwargs
    )
trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
# endregion

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
    prompt_idx = torch.from_numpy(np.array(tokenizer.encode(prompt))).unsqueeze(0)#.to(device)
    prompt_idx = prompt_idx[:, :-1] # remove final token because it is [SEP]
    sample_gen = model.generate(prompt_idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)[0]
    sample_gen = tokenizer.decode(sample_gen)
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

if log_to_wandb:
    samples_table = [[p, g] for p, g in zip(prompts, samples)]
    samples_table = wandb.Table(columns=["Prompt", "Generated Sample"], data = samples_table)
    run.log({"Generated Samples": samples_table})

    wandb.finish()
# endregion