import sys
import argparse
from datetime import datetime

import lightning as L
import torch
import torchinfo
import torchmetrics
import torchvision
import torchvision.transforms as transforms
import wandb
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
parser.add_argument('--activation', default='swiglu', type=str, help='MLP activation')
parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')
parser.add_argument('--norm_first', default=1, type=int, help='whether to use pre-LN or post-LN')
parser.add_argument('--symmetric_rels', default=0, type=int, help='whether to impose symmetric relations in DisRCA')
parser.add_argument('--dff', default=None, type=int, help='feedforward hidden dimension')
parser.add_argument('--patch_size', default=16, type=int, help='size of patches for ViT')
parser.add_argument('--pool', default='cls', type=str, help='type of pooling operation to use')

parser.add_argument('--n_epochs', default=100, type=int, help='number of passes through data to train for')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning_rate')
parser.add_argument('--gradient_accumulation_steps', default=32, type=int, help='gradient_accumulation_steps')
# parser.add_argument('--run_name', default=None, type=str, help='wandb run name')
parser.add_argument('--wandb_project', default='abstract_transformer--Vision-IMAGENET',
    type=str, help='W&B project name')

# configuration of PyTorch Lightning Trainer
parser.add_argument('--eval_interval', default=None, type=int, help='interval of evaluating validation set')
parser.add_argument('--log_every_n_steps', default=None, type=int, help='interval of logging training metrics')
parser.add_argument('--max_steps', default=-1, type=int, help='maximum number of steps')
parser.add_argument('--log_model', default=1, type=int, help='whether to save the model at the end of training')
parser.add_argument('--log_to_wandb', default=1, type=int, help='whether to log to wandb')
parser.add_argument('--compile', default=1, type=int, help='whether to compile')

parser.add_argument('--resume', default=1, type=int, help='whether to resume from a previous run')
parser.add_argument('--ckpt_path', default='NA', type=str, help='path to checkpoint')
parser.add_argument('--run_id', default='NA', type=str, help='W&B run ID for resuming')

args = parser.parse_args()

resume = bool(args.resume)
if resume:
    if args.ckpt_path == 'NA':
        raise ValueError(f'must specify ckpt_path if resume=1. received ckpt_path={args.ckpt_path}')
    if args.run_id == 'NA' and bool(args.log_to_wandb):
        raise ValueError(f'must specify run_id if resume=1. received run_id={args.run_id}')

    ckpt = torch.load(args.ckpt_path)

batch_size = args.batch_size
n_epochs = args.n_epochs

# get model config from args (and fix others)
d_model, sa, rca, n_layers = args.d_model, args.sa, args.rca, args.n_layers
dff = args.dff
rca_type = args.rca_type
symmetric_rels = bool(args.symmetric_rels) if args.symmetric_rels in (0,1) else None
symbol_type = args.symbol_type
dropout_rate = args.dropout_rate
activation = args.activation
norm_first = bool(args.norm_first)
bias = False
patch_size = (args.patch_size, args.patch_size)
pool = args.pool

datetime_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
if rca == 0:
    run_name = f'sa={sa}; d={d_model}; L={n_layers}__{datetime_now}'
else:
    run_name = f'sa={sa}; rca={rca}; d={d_model}; L={n_layers}; rca_type={rca_type}; sym_rel={symmetric_rels}; symbol_type={symbol_type}__{datetime_now}'

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
learning_rate = args.learning_rate # 1e-3 # with baby networks can afford to go a bit higher
# max_iters = 5000
grad_clip = 0.0 # 1.0 # clip gradients at this value, or disable if == 0.0
decay_lr = True # whether to decay the learning rate
# lr_decay_iters = 5000 # make equal to max_iters usually
weight_decay = 1e-1
min_lr = 1e-4 # learning_rate / 10 usually
beta1 = 0.9
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
# warmup_iters = 100
gradient_accumulation_steps = args.gradient_accumulation_steps #1 # accumulate gradients over this many steps. simulates larger batch size

# endregion

# region data set up
# load IMAGENET data
from torch.utils.data import DataLoader
from imagenet_data_utils import ImageNetKaggle

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

normalize = transforms.Normalize(mean=mean,std=std)
train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])


def inv_normalize(tensor, mean=mean, std=std):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


root = '/home/ma2393/scratch/datasets/imagenet'
train_ds = ImageNetKaggle(root, "train", train_transform)
train_dataloader = DataLoader(
            train_ds,
            batch_size=batch_size, # may need to reduce this depending on your GPU 
            num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )

val_ds = ImageNetKaggle(root, "val", val_transform)
val_dataloader = DataLoader(
            val_ds,
            batch_size=batch_size, # may need to reduce this depending on your GPU 
            num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
            shuffle=False,
            drop_last=True,
            pin_memory=True
        )

n_classes = 1000

c, w, h = (3, 224, 224)
image_shape = (c, w, h)

n_patches = (w // patch_size[0]) * (h // patch_size[1])
# endregion

# region define Pytorch Lightning Module

log_on_step = True
topks = 10
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
        acc = self.accuracy(logits, y)

        self.log(f"val/loss", loss, prog_bar=True, logger=True, add_dataloader_idx=False)
        self.log(f"val/acc", acc, prog_bar=True, logger=True, add_dataloader_idx=False)

        for k in range(1, topks):
            acc = torchmetrics.functional.accuracy(logits, y, task="multiclass", num_classes=n_classes, top_k=k, average='micro')
            self.log(f"val/top{k}_acc", acc, prog_bar=True, logger=True, add_dataloader_idx=False)


    def configure_optimizers(self):
        optimizer = configure_optimizers(self.model, weight_decay, learning_rate, (beta1, beta2), device_type=device)
        return optimizer

# endregion


# define model

# define kwargs for symbol-retrieval module based on type
rca_kwargs = dict()
if symbol_type == 'symbolic_attention':
    symbol_retrieval_kwargs = dict(d_model=d_model, n_symbols=n_patches, n_heads=4) # NOTE: n_heads, n_symbols fixed for now
elif symbol_type == 'positional_symbols':
    symbol_retrieval_kwargs = dict(symbol_dim=d_model, max_length=n_patches+1)
elif symbol_type == 'position_relative':
    symbol_retrieval_kwargs = dict(symbol_dim=d_model, max_rel_pos=n_patches+1)
    rca_kwargs['use_relative_positional_symbols'] = True # if using position-relative symbols, need to tell RCA module
elif symbol_type == 'positional_symbols':
    symbol_retrieval_kwargs = dict(symbol_dim=d_model, max_length=n_patches+1)
elif rca != 0:
    raise ValueError(f'`symbol_type` {symbol_type} not valid')

if rca_type == 'disentangled_v2':
    rca_kwargs['symmetric_rels'] = symmetric_rels

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
        activation=activation, norm_first=norm_first, bias=bias, rca_type=rca_type,
        symbol_retrieval=symbol_type, symbol_retrieval_kwargs=symbol_retrieval_kwargs, rca_kwargs=rca_kwargs)

    model = abstracttransformer_lm = VisionDualAttnTransformer(**model_args).to(device)

print(torchinfo.summary(
    model, input_size=(1, *image_shape),
    col_names=("input_size", "output_size", "num_params", "params_percent")))

n_params = sum(p.numel() for p in model.parameters())
print("param count: ", n_params)

# load checkpoint if resume
if resume:
    model_state_dict = {k.split('model.')[1]: v for k,v in ckpt['state_dict'].items()}
    model.load_state_dict(model_state_dict)

# compile model
if args.compile:
    model = torch.compile(model, fullgraph=True, mode='default')

# create LitLanguageModel
lit_model = LitVisionModel(model)
# endregion

# region train model

if log_to_wandb:
    if resume:
        run = wandb.init(project=wandb_project, id=args.run_id, resume='must')
    else:
        run = wandb.init(project=wandb_project, group=group_name, name=run_name,
            config={'group': group_name, 'num_params': n_params, **model_args})

    wandb_logger = WandbLogger(experiment=run, log_model=log_model),
else:
    wandb_logger = None


callbacks = [
    TQDMProgressBar(refresh_rate=50),
    L.pytorch.callbacks.ModelCheckpoint(dirpath=f'out/imagenet/{run_name}', save_top_k=1)
]

trainer_kwargs = dict(
    max_epochs=n_epochs, enable_checkpointing=True, enable_model_summary=False, benchmark=True,
    enable_progress_bar=True, callbacks=callbacks, logger=wandb_logger,
    accumulate_grad_batches=gradient_accumulation_steps, gradient_clip_val=grad_clip,
    log_every_n_steps=log_every_n_steps, max_steps=max_steps, val_check_interval=eval_interval)

trainer = L.Trainer(
    **trainer_kwargs
    )

if args.resume:
    trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=args.ckpt_path)
else:
    trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if log_to_wandb:
    wandb.finish()
# endregion
