import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
import torchinfo
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
import warmup_scheduler
import numpy as np
import sys
import random

from utils import get_model, get_dataset, get_experiment_name
from da import CutMix, MixUp

parser = argparse.ArgumentParser()
parser.add_argument("--wandb_project", default=None, type=str)
parser.add_argument("--wandb_entity", default='dual-attention', type=str)

parser.add_argument("--dataset", default="cifar10", type=str, help="[cifar10, cifar100, svhn]")
parser.add_argument("--num-classes", default=10, type=int)

parser.add_argument('--sa', default=12, type=int, help='number of self-attention heads')
parser.add_argument('--ra', default=0, type=int, help='number of relational attention heads')
parser.add_argument('--symbol_type', default='NA', type=str, choices=('positional_symbols', 'position_relative', 'symbolic_attention', 'NA'), help='type of symbols to use')
parser.add_argument('--ra_type', default='NA', type=str, choices=('relational_attention', 'rca', 'disrca', 'NA'), help="type of RA to use")
parser.add_argument('--n_layers', default=8, type=int, help='number of layers')
parser.add_argument('--d_model', default=384, type=int, help='model dimension')
parser.add_argument('--dff', default=384, type=int, help='feedforward hidden dimension')
parser.add_argument('--activation', default='gelu', type=str, help='MLP activation')
parser.add_argument('--dropout_rate', default=0., type=float, help='dropout rate')
parser.add_argument('--norm_first', default=1, type=int, help='whether to use pre-LN or post-LN')
parser.add_argument('--symmetric_rels', default=0, type=int, help='whether to impose symmetric relations in RA')
parser.add_argument('--n_kv_heads', type=int, default=None, help='Number of key/value heads (e.g., MQA if 1)')
parser.add_argument('--n_relations', default=None, type=int, help='Number of relations in RA')
parser.add_argument('--use_rope', default=0, type=int, help='whether to apply RoPE in attention')
parser.add_argument('--update_symbols_each_layer', default=1, type=int, help='Whether to update symbols each layer')
parser.add_argument('--n_symbols', default=None, type=int, help='Number of symbols in Symbolic Attention')
parser.add_argument('--rel_activation', type=str, default='identity', help='Relation activation function')
parser.add_argument('--share_attn_params', type=int, default=0, help='whether to share wq/wk across SA and RA in DA')
parser.add_argument('--patch_size', default=4, type=int, help='size of patches for ViT')
# parser.add_argument("--patch", default=8, type=int)
parser.add_argument('--pool', default='cls', type=str, help='type of pooling operation to use')
parser.add_argument('--bias', default=1, type=int, help='whether to use bias')

parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=1024, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--min-lr", default=1e-5, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--off-benchmark", action="store_true")
parser.add_argument("--max-epochs", default=100, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight-decay", default=5e-5, type=float)
parser.add_argument("--warmup-epoch", default=5, type=int)
parser.add_argument("--precision", default='bf16-mixed', type=str)
parser.add_argument("--autoaugment", action="store_true")
parser.add_argument("--criterion", default="ce")
parser.add_argument("--label-smoothing", action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--compile", action="store_true")

parser.add_argument("--seed", default=None, type=int)
args = parser.parse_args()

if args.seed is None:
    print("Seed not specified by script arguments. Will generate randomly.")
    args.seed = random.randrange(2**32 - 1)

print(f"Setting seed to: {args.seed}")
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# process args
args.model_name = 'vidat' if args.ra != 0 else 'vit'
args.norm_first = True if args.norm_first==1 else False
args.symmetric_rels = True if args.symmetric_rels==1 else False
args.share_attn_params = True if args.share_attn_params==1 else False
args.patch_size = (args.patch_size, args.patch_size)
args.bias = args.bias
args.update_symbols_each_layer = (args.update_symbols_each_layer == 1)
args.use_rope = (args.use_rope == 1)
args.rel_proj_dim = None if args.n_relations is None else int((args.d_model / (args.sa+args.ra)) * (args.ra / args.n_relations))
if args.model_name == 'vidat' and (args.ra_type == 'NA' or args.symbol_type == 'NA'):
    raise ValueError(f'RA type and symbol type must be specified for ViDAT, got {args.ra_type} and {args.symbol_type}')

args.benchmark = True if not args.off_benchmark else False
args.gpus = torch.cuda.device_count()
args.num_workers = 4*args.gpus if args.gpus else 8
args.is_cls_token = args.pool == 'cls'
if not args.gpus:
    args.precision=32

if args.dff != args.d_model*4:
    print(f"[INFO] In original paper, dff (CURRENT:{args.dff}) is set to: {args.d_model*4}(={args.d_model}*4)")

train_ds, test_ds = get_dataset(args)
args.image_shape = (args.in_c, args.size, args.size)
n_patches = (args.size // args.patch_size[0]) * (args.size // args.patch_size[1])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)

# parse DAT config args
if args.ra != 0:
    args.ra_kwargs = dict(n_relations=args.n_relations, rel_activation=args.rel_activation, rel_proj_dim=args.rel_proj_dim, n_kv_heads=args.n_kv_heads)
    args.sa_kwargs = dict(n_kv_heads=args.n_kv_heads)

    if args.symbol_type == 'symbolic_attention':
        n_symbols = args.n_symbols if args.n_symbols is not None else n_patches
        args.symbol_retrieval_kwargs = dict(d_model=args.d_model, n_symbols=n_symbols, n_heads=4) # NOTE: n_heads, n_symbols fixed for now
    elif args.symbol_type == 'positional_symbols':
        args.symbol_retrieval_kwargs = dict(symbol_dim=args.d_model, max_length=n_patches+1)
    elif args.symbol_type == 'position_relative':
        args.symbol_retrieval_kwargs = dict(symbol_dim=args.d_model, max_rel_pos=n_patches+1)
        args.ra_kwargs['use_relative_positional_symbols'] = True # if using position-relative symbols, need to tell RA module
    elif args.symbol_type == 'positional_symbols':
        args.symbol_retrieval_kwargs = dict(symbol_dim=args.d_model, max_length=n_patches+1)
    elif args.ra != 0:
        raise ValueError(f'`symbol_type` {args.symbol_type} not valid')

    if args.ra_type == 'relational_attention':
        args.ra_kwargs['symmetric_rels'] = args.symmetric_rels


topks = 10
class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        # self.hparams = hparams
        self.hparams.update(vars(hparams))
        self.model = get_model(hparams)
        if args.compile:
            self.model = torch.compile(self.model)
        # self.criterion = get_criterion(args)
        if hparams.label_smoothing:
            self.criterion = nn.CrossEntropyLoss(args.num_classes, label_smoothing=args.smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
        if hparams.cutmix:
            self.cutmix = CutMix(hparams.size, beta=1.)
        if hparams.mixup:
            self.mixup = MixUp(alpha=1.)
        self.log_image_flag = hparams.wandb_project is None

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)

        self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=self.hparams.warmup_epoch, after_scheduler=self.base_scheduler)

        # TODO: switch to using ignite...
        # self.scheduler = ignite.handlers.param_scheduler.create_lr_scheduler_with_warmup(self.base_scheduler,
            # warmup_start_value=self.hparams.lr*0.1, warmup_end_value=self.hparams.lr, warmup_duration=self.hparams.warmup_epoch)

        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        img, label = batch
        if self.hparams.cutmix or self.hparams.mixup:
            if self.hparams.cutmix:
                img, label, rand_label, lambda_= self.cutmix((img, label))
            elif self.hparams.mixup:
                if np.random.rand() <= 0.8:
                    img, label, rand_label, lambda_ = self.mixup((img, label))
                else:
                    img, label, rand_label, lambda_ = img, label, torch.zeros_like(label), 1.
            out = self.model(img)
            loss = self.criterion(out, label)*lambda_ + self.criterion(out, rand_label)*(1.-lambda_)
        else:
            out = self(img)
            loss = self.criterion(out, label)

        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("loss/train", loss)
        self.log("acc/train", acc)
        return loss

    def on_train_epoch_end(self):
        self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=True)#self.current_epoch)

    def validation_step(self, batch, batch_idx):
        img, label = batch
        logits = self(img)
        loss = self.criterion(logits, label)
        acc = torch.eq(logits.argmax(-1), label).float().mean()
        self.log("loss/val", loss)
        self.log("acc/val", acc)

        for k in range(1, topks+1):
            acc = torchmetrics.functional.accuracy(logits, label, task="multiclass", num_classes=args.num_classes, top_k=k, average='micro')
            self.log(f"topk/top{k}_valacc", acc, prog_bar=True, logger=True, add_dataloader_idx=False)


        return loss


if __name__ == "__main__":
    experiment_name, run_name = get_experiment_name(args)
    print(f"experiment name: {experiment_name}")

    if args.wandb_project is not None:
        print("[INFO] Log with WandB!")
        logger = pl.loggers.WandbLogger(
            project=args.wandb_project,
            name=run_name,
            group=experiment_name,
            entity=args.wandb_entity,
            config=args
        )
        refresh_rate = 0

    else:
        print("[INFO] Log with CSV")
        logger = pl.loggers.CSVLogger(
            save_dir="logs",
            name=experiment_name
        )
        refresh_rate = 1
    net = Net(args)

    model_summary = torchinfo.summary(net.model, input_size=(1, *args.image_shape),
        col_names=("input_size", "output_size", "num_params", "params_percent"))
    print(model_summary)

    model_summary_dict = {
        'Input size (MB)': model_summary.to_megabytes(model_summary.total_input),
        'Params size (MB)': model_summary.to_megabytes(model_summary.total_param_bytes),
        'Forward/backward pass size  (MB)': model_summary.to_megabytes(model_summary.total_output_bytes),
        'Estimated total size (MB)': model_summary.to_megabytes(model_summary.total_output_bytes + model_summary.total_param_bytes + model_summary.total_input),
        'Total Mult-Adds': model_summary.total_mult_adds,

        'trainable_params': model_summary.trainable_params, # note: numbers from torchinfo are not always accurate
        'total_params': model_summary.total_params, # note: numbers from torchinfo are not always accurate

        'num_params': sum(p.numel() for p in net.model.parameters()),
        'num_trainable_params': sum(p.numel() for p in net.model.parameters() if p.requires_grad)
    }

    print(f'num params: {model_summary_dict["num_params"]:,}')
    print(f'num trainable params: {model_summary_dict["num_trainable_params"]:,}')
    args.model_summary = model_summary_dict

    trainer = pl.Trainer(precision=args.precision,fast_dev_run=args.dry_run, devices=args.gpus, benchmark=args.benchmark, logger=logger, max_epochs=args.max_epochs, callbacks=[TQDMProgressBar(refresh_rate=100)])

    trainer.fit(model=net, train_dataloaders=train_dl, val_dataloaders=test_dl)

    # if not args.dry_run:
    #     model_path = f"model_checkpoints/{experiment_name}.pth"
    #     torch.save(net.state_dict(), model_path)
        # if args.wandb_project is not None:
        #     logger.experiment.log_asset(file_name=experiment_name, file_data=model_path)