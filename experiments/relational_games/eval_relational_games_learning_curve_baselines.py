import sys
import argparse
from datetime import datetime

import numpy as np
import torch

import lightning as L
import torchinfo
import torchmetrics

import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

from relational_games_data_utils import RelationalGamesDataset
from baseline_models import CoRelNet, CoRelNetSoftmax, PrediNet, AbstractorModel


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


# region parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--task', required=True, type=str, help='relational games task')
parser.add_argument('--model', required=True, choices=('corelnet', 'corelnet_softmax', 'predinet', 'abstractor'))

parser.add_argument('--patch_size', default=12, type=int, help='size of image patches')

parser.add_argument('--train_sizes', nargs='+', required=True, type=int, help='training set sizes for learning curves')
parser.add_argument('--n_trials', default=1, type=int, help='training set sizes for learning curves')
parser.add_argument('--val_size', default=5_000, type=int, help='validation set size')
parser.add_argument('--test_size', default=10_000, type=int, help='test set size')
parser.add_argument('--n_epochs', default=50, type=int, help='number of passes through data to train for')
parser.add_argument('--batch_size', default=512, type=int, help='batch size')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wandb_entity', default='awni00')
parser.add_argument('--wandb_project', default='dual_attention--relational_games_learning_curves',
    type=str, help='W&B project name')

# configuration of PyTorch Lightning Trainer
parser.add_argument('--early_stopping', default=0, type=int, help='whether to use early stopping')
parser.add_argument('--eval_interval', default=None, type=int, help='interval of evaluating validation set')
parser.add_argument('--log_every_n_steps', default=1, type=int, help='interval of logging training metrics')
parser.add_argument('--max_steps', default=-1, type=int, help='maximum number of steps')
parser.add_argument('--log_model', default=0, type=int, help='whether to save the model at the end of training')
parser.add_argument('--log_to_wandb', default=1, type=int, help='whether to log to wandb')
parser.add_argument('--compile', default=1, type=int, help='whether to compile model')
parser.add_argument('--precision', default='bf16', type=str, help='precision arg to Trainer')
args = parser.parse_args()

task = args.task
batch_size = args.batch_size
n_epochs = args.n_epochs
n_trials = args.n_trials

# get model config from args (and fix others)
model_type = args.model
patch_size = (args.patch_size, args.patch_size)

# run_name = args.run_name if args.run_name is not None else group_name
wandb_project = args.wandb_project
log_to_wandb = bool(args.log_to_wandb)
train_sizes = args.train_sizes
val_size = args.val_size
test_size = args.test_size

eval_interval, max_steps, log_every_n_steps = args.eval_interval, args.max_steps, args.log_every_n_steps
log_on_step = True # log metrics of training steps (at eval_interval)
# endregion

# region some configuration
device = 'cuda'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# # optimization hyperparams
learning_rate = args.learning_rate
grad_clip = 0.0 # 1.0 # clip gradients at this value, or disable if == 0.0
beta1 = 0.9
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
gradient_accumulation_steps = 1 # accumulate gradients over this many steps. simulates larger batch size
early_stopping = bool(args.early_stopping)

# endregion

# region data set up
data_path = '../../data/relational_games'

train_split = 'pentos'
train_ds = RelationalGamesDataset(data_path, task, train_split)

eval_ds_dict = dict()
eval_dls = []
val_splits = ['hexos', 'stripes']
for val_split in val_splits:
    ds = RelationalGamesDataset(data_path, task, val_split)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    eval_ds_dict[val_split] = ds
    eval_dls.append(dl)

val_splits.append('in_distribution')

c, w, h = (3, 36, 36)
image_shape = (c, w, h)
n_classes = 2

n_patches = (w // patch_size[0]) * (h // patch_size[1])
# endregion

# region define Pytorch Lightning Module
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

        self.log('train/loss', loss)
        self.log('train/acc', acc)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)

        self.log(f"val/loss", loss)
        self.log(f"val/acc", acc)

    def test_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)

        self.log(f"test/loss_{val_splits[dataloader_idx]}", loss, add_dataloader_idx=False)
        self.log(f"test/acc_{val_splits[dataloader_idx]}", acc, add_dataloader_idx=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(beta1, beta2))
        return optimizer

# endregion

# region define model

# define kwargs for symbol-retrieval module based on type

if model_type == 'corelnet':
    group_name = f'{task}__corelnet'
    model_config = dict(image_shape=(3, 36, 36), patch_size=(12, 12), d_model=128, num_classes=2)
    def create_model():
        corelnet = CoRelNet(**model_config)
        return corelnet
elif model_type == 'corelnet_softmax':
    group_name = f'{task}__corelnet_softmax'
    model_config = dict(image_shape=(3, 36, 36), patch_size=(12, 12), d_model=128, num_classes=2)
    def create_model():
        corelnet = CoRelNetSoftmax(**model_config)
        return corelnet
elif model_type == 'predinet':
    group_name = f'{task}__predinet'
    model_config = dict(image_shape=(3, 36, 36), patch_size=(12, 12), d_model=128, n_heads=16, n_relations=8, n_classes=2)
    def create_model():
        predinet = PrediNet(**model_config)
        return predinet
elif model_type == 'abstractor':
    group_name = f'{task}__abstractor'
    model_config = dict(image_shape=(3, 36, 36), patch_size=(12, 12), d_model=128,
        abstractor_kwargs=dict(n_layers=2, n_heads=4, dff=128*2, use_self_attn=True, dropout_rate=0, norm_first=True, activation='relu',
            symbol_retriever_type='positional', symbol_retriever_kwargs=dict(symbol_dim=128, max_length=9), symbol_add_pos_embedding=False, max_len=9),
        num_classes=2)
    def create_model():
        abstractor = AbstractorModel(**model_config)
        return abstractor

model_args = dict(baseline_model=args.model, model_config=model_config)


# endregion

# region eval learning curves
for trial in range(n_trials):
    for train_size in train_sizes:
        datetime_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        datetime_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_name = f'{group_name}_train_size={train_size}__{datetime_now}'

        # randomly sample a training split and validation split based on train size
        subsets_sample = np.random.choice(len(train_ds), train_size+val_size+test_size)
        train_subset = subsets_sample[:train_size]
        val_subset = subsets_sample[train_size:train_size+val_size]
        test_subset = subsets_sample[train_size+val_size:]

        train_sample_ds = torch.utils.data.Subset(train_ds, train_subset)
        val_sample_ds = torch.utils.data.Subset(train_ds, val_subset)
        test_sample_ds = torch.utils.data.Subset(train_ds, test_subset)

        # create data loaders
        train_dataloader = torch.utils.data.DataLoader(train_sample_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(val_sample_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(test_sample_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        model = create_model()

        n_params = sum(p.numel() for p in model.parameters())

        # compile model
        if args.compile:
            print('compiling model...')
            model = torch.compile(model) #, backend='inductor', fullgraph=False, mode='default', dynamic=True)
            print('compiled.')

        # create LitLanguageModel
        lit_model = LitVisionModel(model)

        if args.log_to_wandb:
            run = wandb.init(project=wandb_project, entity=args.wandb_entity, group=group_name, name=run_name,
                config={'group': group_name, 'num_params': n_params, 'task': args.task, 'train_size': train_size, 'val_size': val_size, **model_args})


        callbacks = [
            TQDMProgressBar(),
        ]
        if early_stopping:
            ckpt_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=f'out/{task}/{run_name}', save_top_k=1, monitor="val/loss", mode="min")
            callbacks.append(ckpt_callback)

        trainer_kwargs = dict(
            max_epochs=n_epochs, enable_checkpointing=early_stopping, enable_model_summary=True, benchmark=True,
            enable_progress_bar=True, callbacks=callbacks, logger=False,
            accumulate_grad_batches=gradient_accumulation_steps, gradient_clip_val=grad_clip,
            log_every_n_steps=log_every_n_steps, max_steps=max_steps, check_val_every_n_epoch=1,
            precision=args.precision)

        trainer = L.Trainer(
            **trainer_kwargs
            )
        trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # reload best model
        if early_stopping:
            state_dict = torch.load(ckpt_callback.best_model_path)['state_dict']
            state_dict = {k[len('model.'):]: v for k,v in state_dict.items()}
            lit_model.model.load_state_dict(state_dict)

        eval_results = trainer.test(lit_model, [*eval_dls, test_dataloader])

        eval_results = {k: v for eval_dict in eval_results for k,v in eval_dict.items()}

        if log_to_wandb:
            run.log(eval_results)
            run.finish()


if log_to_wandb:
    wandb.finish()
# endregion
