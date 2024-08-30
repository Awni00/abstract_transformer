import argparse

# import comet_ml # FIXME
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchinfo
import pytorch_lightning as pl
import warmup_scheduler
import numpy as np
import sys

from utils import get_model, get_dataset, get_experiment_name
from da import CutMix, MixUp

parser = argparse.ArgumentParser()
# parser.add_argument("--api-key", help="API Key for Comet.ml")
parser.add_argument("--wandb_project", default=None, type=str)
parser.add_argument("--wandb_entity", default='dual-attention', type=str)
parser.add_argument("--dataset", default="c10", type=str, help="[c10, c100, svhn]")
parser.add_argument("--num-classes", default=10, type=int)
parser.add_argument("--model-name", default="vit", help="[vit]", type=str)
parser.add_argument("--patch", default=8, type=int)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=1024, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--min-lr", default=1e-5, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--off-benchmark", action="store_true")
parser.add_argument("--max-epochs", default=200, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight-decay", default=5e-5, type=float)
parser.add_argument("--warmup-epoch", default=5, type=int)
parser.add_argument("--precision", default='16', type=str)
parser.add_argument("--autoaugment", action="store_true")
parser.add_argument("--criterion", default="ce")
parser.add_argument("--label-smoothing", action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--compile", action="store_true")
parser.add_argument("--dropout", default=0.0, type=float)
parser.add_argument("--head", default=12, type=int)
parser.add_argument("--num-layers", default=7, type=int)
parser.add_argument("--hidden", default=384, type=int)
parser.add_argument("--mlp-hidden", default=384, type=int)
parser.add_argument("--off-cls-token", action="store_true")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--project-name", default="VisionTransformer")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

args.benchmark = True if not args.off_benchmark else False
args.gpus = torch.cuda.device_count()
args.num_workers = 4*args.gpus if args.gpus else 8
args.is_cls_token = True if not args.off_cls_token else False
if not args.gpus:
    args.precision=32

if args.mlp_hidden != args.hidden*4:
    print(f"[INFO] In original paper, mlp_hidden(CURRENT:{args.mlp_hidden}) is set to: {args.hidden*4}(={args.hidden}*4)")

train_ds, test_ds = get_dataset(args)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)

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

        if not self.log_image_flag and not self.hparams.dry_run:
            self.log_image_flag = True
            self._log_image(img.clone().detach().cpu())

        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("loss", loss)
        self.log("acc", acc)
        return loss

    def on_train_epoch_end(self):
        self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=True)#self.current_epoch)

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def _log_image(self, image):
        grid = torchvision.utils.make_grid(image, nrow=4)
        # self.logger.log_image(key='examples', images=grid.permute(1,2,0)) # FIXME
        print("[INFO] LOG IMAGE!!!")


if __name__ == "__main__":
    experiment_name = get_experiment_name(args)
    print(experiment_name)
    # if args.api_key:
    #     print("[INFO] Log with Comet.ml!")
    #     logger = pl.loggers.CometLogger(
    #         api_key=args.api_key,
    #         save_dir="logs",
    #         project_name=args.project_name,
    #         experiment_name=experiment_name
    #     )
    #     refresh_rate = 0
    if args.wandb_project is not None:
        print("[INFO] Log with WandB!")
        logger = pl.loggers.WandbLogger(
            project=args.wandb_project,
            name=experiment_name,
            entity=args.wandb_entity
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
    print(torchinfo.summary(net.model, (args.batch_size, args.in_c, args.size, args.size)))

    trainer = pl.Trainer(precision=args.precision,fast_dev_run=args.dry_run, devices=args.gpus, benchmark=args.benchmark, logger=logger, max_epochs=args.max_epochs)
    trainer.fit(model=net, train_dataloaders=train_dl, val_dataloaders=test_dl)
    if not args.dry_run:
        model_path = f"weights/{experiment_name}.pth"
        torch.save(net.state_dict(), model_path)
        if args.wandb_project is not None:
            logger.experiment.log_asset(file_name=experiment_name, file_data=model_path)