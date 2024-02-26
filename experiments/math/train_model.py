import torch
import lightning as L
from lightning.pytorch.loggers.wandb import WandbLogger
import torchmetrics

import numpy as np
import wandb
import torchinfo
from contextlib import nullcontext
from  tqdm import tqdm, trange
import argparse

import os
import sys; sys.path += ['../', '../..']
from seq2seq_models import Seq2SeqAbstractTransformer, Seq2SeqTransformer

# region config
parser = argparse.ArgumentParser()

parser.add_argument('--task', required=True, type=str, help='task name')
parser.add_argument('--ee', required=True, type=int, help='number of encoder self-attention heads')
parser.add_argument('--ea', required=True, type=int, help='number of encoder relational cross-attention heads')
parser.add_argument('--de', required=True, type=int, help='number of decoder self-attention heads')
parser.add_argument('--da', required=True, type=int, help='number of decoder relational cross-attention heads')
parser.add_argument('--e_n_layers', required=True, type=int, help='number of encoder layers')
parser.add_argument('--d_n_layers', required=True, type=int, help='number of decoder layers')

parser.add_argument('--n_epochs', default=50, type=int, help='number of epochs to train each model for')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--run_name', default=None, type=str, help='wandb run name')
parser.add_argument('--wandb_project_prefix', default='abstract_transformer--math',
    type=str, help='W&B project name')
args = parser.parse_args()

batch_size = args.batch_size
n_epochs = args.n_epochs
task = args.task

ee, ea, de, da = args.ee, args.ea, args.de, args.da
e_n_layers = args.e_n_layers
d_n_layers = args.d_n_layers

group_name = f'ee={ee}; ea={ea}; de={de}; da={da}; el={e_n_layers}; dl={d_n_layers}'
run_name = args.run_name

# region some configuration
device = 'cuda'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# dtype = 'float32'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# wandb logging
wandb_log = False
wandb_project = f'{args.wandb_project_prefix}--{task}'
# endregion

# region data
# vocab
with open('text_vectorizer/vocabulary.txt') as f:
    vocab = f.read().splitlines()

idx_to_char = {i: c for i, c in enumerate(vocab)}
char_to_idx = {c: i for i, c in enumerate(vocab)}

empty_token = char_to_idx['']
eos_token = char_to_idx[';']
start_token = char_to_idx['@']

vocab_size = len(vocab)

max_q_len, max_a_len = 161, 31

# load tokenized data
data_path = 'tokenized_data'

train_ds = torch.load(f'{data_path}/{task}_train.pt')
interp_ds = torch.load(f'{data_path}/{task}_interpolate.pt')
extrap_ds = torch.load(f'{data_path}/{task}_interpolate.pt')

# create data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
interp_dl = torch.utils.data.DataLoader(interp_ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
extrap_dl = torch.utils.data.DataLoader(extrap_ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
# endregion

# region training set up
def compute_tf_acc(logits, idx_label, ignore_index=None):
    pred = torch.argmax(logits, dim=-1)
    if ignore_index is None:
        mask = torch.ones_like(idx_label, dtype=torch.bool)
    else:
        mask = idx_label != ignore_index

    match = (idx_label == pred) & mask

    correct_preds = torch.sum(match)
    mask_count = torch.sum(mask)
    return correct_preds / mask_count

class LitSeq2SeqModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y, z = batch
        # with ctx:
        logits, loss = self.model(x, y, z)

        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, z = batch
        # with ctx:
        logits, loss = self.model(x, y, z)

        tf_acc = compute_tf_acc(logits, z, ignore_index=empty_token)
        perplexity = torchmetrics.functional.text.perplexity(logits, z, ignore_index=empty_token)

        prefix = ['interpolate', 'extrapolate'][dataloader_idx]
        self.log(f"{prefix}_loss", loss, prog_bar=True, logger=True, add_dataloader_idx=False)
        self.log(f'{prefix}_teacher_forcing_acc', tf_acc, prog_bar=True, logger=True, add_dataloader_idx=False)
        self.log(f'{prefix}_perplexity', perplexity, prog_bar=True, logger=True, add_dataloader_idx=False)

    def test_step(self, batch, batch_idx):
        x, y, z = batch

        n, seqs_length = y.shape
        output = torch.zeros(size=(n, (seqs_length+1)), dtype=torch.int, device=device)
        output[:,0] = start_token

        for i in range(seqs_length):
            with ctx:
                predictions, _ = self.model(x, output[:, :-1], z)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)
            output[:,i+1] = predicted_id

        elementwise_acc = torch.mean((output[:,1:] == z).float()).item()
        # acc_per_position = [torch.mean((output[:, i+1] == labels_test[:, i]).float()).item() for i in range(seqs_length)]
        seq_acc = torch.mean((torch.all(output[:,1:]==z, axis=1)).float()).item()

        with ctx:
            tf_pred, loss = self.model(x, y, z)
            tf_pred = torch.argmax(tf_pred, axis=-1)
        teacher_forcing_acc = torch.mean((z==tf_pred).float()).item()

        self.log("test_loss", loss)
        self.log("teacher_forcing_acc", teacher_forcing_acc)
        self.log("elementwise_acc", elementwise_acc)
        self.log("seq_acc", seq_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=6e-4, betas=(0.9, 0.995))
        return optimizer
# endregion

# region build model
d_model = 128
dff = 256

model_args = dict(
    input_spec=dict(type='token', vocab_size=vocab_size), output_spec=dict(type='token', vocab_size=vocab_size),
    symbol_retrieval='sym_attn', symbol_retrieval_kwargs=dict(model_dim=d_model, n_heads=4, num_symbols=256, dropout=0.0),
    d_model=d_model, out_dim=vocab_size, n_layers_enc=e_n_layers, n_layers_dec=d_n_layers,
    encoder_kwargs=dict(n_heads_enc=ee, n_heads_abs=ea, dff=dff, activation='relu', norm_first=False, dropout_rate=0.1, causal=False, rel_mask_diag=False),
    decoder_kwargs=dict(n_heads_enc=de, n_heads_abs=da, n_heads_cross=2, dff=dff, activation='relu', norm_first=False, dropout_rate=0.1, causal=True, rel_mask_diag=False),
    in_block_size=max_q_len, out_block_size=max_a_len)
model = Seq2SeqAbstractTransformer(**model_args)#.to(device)
torchinfo.summary(model, row_settings=["depth", "var_names"], col_names=["num_params", "params_percent", "trainable"], depth=3, col_width=20)

lit_model = LitSeq2SeqModel(model)
# endregion

# region train
run = wandb.init(project=wandb_project, group=group_name, name=run_name,
    config={'group': group_name, **model_args})

wandb_logger = WandbLogger(experiment=run, log_model=False)
callbacks = [
    L.pytorch.callbacks.TQDMProgressBar(refresh_rate=1000)
]
# wandb_logger.watch(model, log_graph=False)
trainer = L.Trainer(
    max_epochs=n_epochs, enable_checkpointing=False, enable_model_summary=True, # precision='64-true',
    callbacks=callbacks, logger=wandb_logger, enable_progress_bar=True, check_val_every_n_epoch=5,
    )
trainer.fit(model=lit_model, train_dataloaders=train_dl, val_dataloaders=[interp_dl, extrap_dl])

wandb.finish()
# endregion