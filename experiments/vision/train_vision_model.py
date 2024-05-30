import os
print(os.environ['CONDA_DEFAULT_ENV'])
print(os.environ["CONDA_PREFIX"])

import sys
import argparse

import torch
import torchinfo
import torchmetrics
import torchvision
import torchvision.transforms as transforms
import wandb
import lightning as L
from lightning.pytorch.loggers.wandb import WandbLogger

sys.path.append('../..')
from utils.pl_tqdm_progbar import TQDMProgressBar
from vision_models import VisionDualAttnTransformer, VisionTransformer, configure_optimizers

# print cuda information
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
parser.add_argument('--symbol_type', required=True, type=str, choices=('positional_symbols', 'position_relative', 'symbolic_attention', 'NA'), help='type of symbols to use')
parser.add_argument('--rca_type', required=True, type=str, choices=('relational_attention', 'rca', 'disrca', 'NA'), help="type of RCA to use")
parser.add_argument('--n_layers', required=True, type=int, help='number of layers')
parser.add_argument('--d_model', required=True, type=int, help='model dimension')
parser.add_argument('--patch_size', required=True, type=int, help='size of patches for ViT')
parser.add_argument('--pool', default='cls', type=str, help='type of pooling operation to use')
# parser.add_argument('--dff', required=True, type=int, help='feedforward hidden dimension')

parser.add_argument('--n_epochs', default=1, type=int, help='number of passes through data to train for')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
# parser.add_argument('--run_name', default=None, type=str, help='wandb run name')
parser.add_argument('--wandb_project', default='abstract_transformer--Vision-CIFAR10',
    type=str, help='W&B project name')

# configuration of PyTorch Lightning Trainer
parser.add_argument('--eval_interval', default=None, type=int, help='interval of evaluating validation set')
parser.add_argument('--log_every_n_steps', default=None, type=int, help='interval of logging training metrics')
parser.add_argument('--max_steps', default=-1, type=int, help='maximum number of steps')
parser.add_argument('--log_model', default=1, type=int, help='whether to save the model at the end of training')
parser.add_argument('--log_to_wandb', default=1, type=int, help='whether to log to wandb')
parser.add_argument('--compile', default=0, type=int, help='whether to compile model')
args = parser.parse_args()

batch_size = args.batch_size
n_epochs = args.n_epochs

# get model config from args (and fix others)
d_model, sa, rca, n_layers = args.d_model, args.sa, args.rca, args.n_layers
dff = None
rca_type = args.rca_type
symbol_type = args.symbol_type
dropout_rate = 0.2
activation = 'gelu' # gelu rather than relu
norm_first = True
bias = True
patch_size = (args.patch_size, args.patch_size)
pool = args.pool

run_name = f'sa={sa}; rca={rca}; d={d_model}; L={n_layers}; rca_type={rca_type}; symbol_type={symbol_type}'
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
# max_iters = 5000
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
decay_lr = True # whether to decay the learning rate
# lr_decay_iters = 5000 # make equal to max_iters usually
weight_decay = 1e-1
min_lr = 1e-4 # learning_rate / 10 usually
beta1 = 0.9
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
# warmup_iters = 100
gradient_accumulation_steps = 1 # accumulate gradients over this many steps. simulates larger batch size

# endregion

# region data set up
# load CIFAR10 data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
n_classes = 10
assert n_classes == len(classes)

dataiter = iter(train_dataloader)
images, labels = next(dataiter)
c, w, h = images.shape[1:]
image_shape = (c, w, h)

n_patches = (w // patch_size[0]) * (h // patch_size[1])
# endregion

# region define Pytorch Lightning Module

log_on_step = True
class LitVisionModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.functional.cross_entropy
        self.accuracy = lambda pred, y: torchmetrics.functional.accuracy(pred, y, task="multiclass", num_classes=n_classes, top_k=1, average='micro')

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)

        self.log('train/loss', loss, prog_bar=True, logger=True, on_step=log_on_step, on_epoch=True)
        self.log('train/acc', acc, prog_bar=True, logger=True, on_step=log_on_step, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        # acc = torchmetrics.functional.accuracy(logits, y, task="multiclass", num_classes=n_classes, top_k=1, average='micro')
        acc = self.accuracy(logits, y)

        self.log("val/loss", loss, prog_bar=True, logger=True, add_dataloader_idx=False)
        self.log("val/acc", acc, prog_bar=True, logger=True, add_dataloader_idx=False)

    def configure_optimizers(self):
        optimizer = configure_optimizers(self.model, weight_decay, learning_rate, (beta1, beta2), device_type=device)
        return optimizer

# endregion

# define model

# define kwargs for symbol-retrieval module based on type
rca_kwargs = dict()
if symbol_type == 'symbolic_attention':
    symbol_retrieval_kwargs = dict(d_model=d_model, n_symbols=50, n_heads=4) # NOTE: n_heads, n_symbols fixed for now
elif symbol_type == 'positional_symbols':
    symbol_retrieval_kwargs = dict(symbol_dim=d_model, max_length=n_patches+1)
elif symbol_type == 'position_relative':
    symbol_retrieval_kwargs = dict(symbol_dim=d_model, max_rel_pos=n_patches+1)
    rca_kwargs['use_relative_positional_symbols'] = True # if using position-relative symbols, need to tell RCA module
elif symbol_type == 'positional_symbols':
    symbol_retrieval_kwargs = dict(symbol_dim=d_model, max_length=n_patches+1)
elif rca != 0:
    raise ValueError(f'`symbol_type` {symbol_type} not valid')

# if rca=0, use TransformerLM
if rca == 0:
    model_args = dict(
        image_shape=image_shape, patch_size=patch_size, num_classes=n_classes, pool=pool,
        d_model=d_model, n_layers=n_layers, n_heads=sa, dff=dff, dropout_rate=dropout_rate,
        activation=activation, norm_first=norm_first, bias=bias)

    model = transformer_lm = VisionTransformer(**model_args).to(device)
# otherwise, use AbstractTransformerLM
else:
    model_args = dict(
        image_shape=image_shape, patch_size=patch_size, num_classes=n_classes, pool=pool,
        d_model=d_model, n_layers=n_layers, n_heads_sa=sa, n_heads_rca=rca, dff=dff, dropout_rate=dropout_rate,
        activation=activation, norm_first=norm_first, bias=bias,
        symbol_retrieval=symbol_type, symbol_retrieval_kwargs=symbol_retrieval_kwargs, rca_type=rca_type, rca_kwargs=rca_kwargs)

    model = abstracttransformer_lm = VisionDualAttnTransformer(**model_args).to(device)

print(torchinfo.summary(
    model, input_size=(1, *image_shape),
    col_names=("input_size", "output_size", "num_params", "params_percent")))

n_params = sum(p.numel() for p in model.parameters())
print("param count: ", n_params)

# compile model
if args.compile:
    print('compiling model...')
    model = torch.compile(model) #, backend='inductor', fullgraph=False, mode='default', dynamic=True)
    print('compiled.')

# create LitLanguageModel
lit_model = LitVisionModel(model)
# endregion

# region train model

if log_to_wandb:
    run = wandb.init(project=wandb_project, group=group_name, name=run_name,
        config={'group': group_name, 'num_params': n_params, **model_args})

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


if log_to_wandb:
    wandb.finish()
# endregion
