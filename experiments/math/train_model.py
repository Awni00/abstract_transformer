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
from seq2seq_models import Seq2SeqDualAttnTransformer, Seq2SeqTransformer

# region config
parser = argparse.ArgumentParser()

parser.add_argument('--task', required=True, type=str, help='task name')
parser.add_argument('--e_sa', required=True, type=int, help='number of encoder self-attention heads')
parser.add_argument('--e_ra', required=True, type=int, help='number of encoder relational attention heads')
parser.add_argument('--d_sa', required=True, type=int, help='number of decoder self-attention heads')
parser.add_argument('--d_ra', required=True, type=int, help='number of decoder relational attention heads')
parser.add_argument('--d_cross', required=True, type=int, help='number of decoder cross-attention heads')
parser.add_argument('--symbol_type', required=True, type=str, choices=('position_relative', 'symbolic_attention', 'position_relative', 'NA'), help='type of symbols to use')
parser.add_argument('--update_symbols_each_layer', default=1, type=int, help='whether to update symbols each layer')
parser.add_argument('--ra_type', required=True, type=str, choices=('relational_attention', 'rca', 'disrca', 'NA'), help="type of relational attn to use")
parser.add_argument('--pos_enc_type', default='sinusoidal', type=str, choices=('sinusoidal', 'learned', 'RoPE'), help='type of positional encoding to use')
parser.add_argument('--e_n_layers', required=True, type=int, help='number of encoder layers')
parser.add_argument('--d_n_layers', required=True, type=int, help='number of decoder layers')
parser.add_argument('--d_model', required=True, type=int, help='model dimension')
parser.add_argument('--activation', default='relu', type=str, help='MLP activation')
parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')
parser.add_argument('--dff', required=True, type=int, help='feedforward hidden dimension')
parser.add_argument('--learning_rate', default=6e-4, help='learning rate')

parser.add_argument('--n_epochs', default=50, type=int, help='number of epochs to train each model for')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--shuffle', default=1, type=int, help='whether to shuffle train loader (0 or 1)')
parser.add_argument('--train_postfix', default='train', help='training split (i.e,, train-easy, train-medium, train-hard, train)')
parser.add_argument('--run_name', default=None, type=str, help='wandb run name')
parser.add_argument('--wandb_project_prefix', default='dual_attention--math',
    type=str, help='W&B project name')
args = parser.parse_args()

batch_size = args.batch_size
n_epochs = args.n_epochs
task = args.task

e_sa, e_ra, d_sa, d_ra, d_cross = args.e_sa, args.e_ra, args.d_sa, args.d_ra, args.d_cross
e_n_layers = args.e_n_layers
d_n_layers = args.d_n_layers
d_model = args.d_model
dff = args.dff
ra_type = args.ra_type
activation = args.activation
symbol_type = args.symbol_type
update_symbols_each_layer = (args.update_symbols_each_layer == 1)
dropout_rate  = args.dropout_rate
pos_enc_type = args.pos_enc_type

group_name = f'enc_sa={e_sa}; enc_ra={e_ra}; dec_sa={d_sa}; dec_ra={d_ra}; d_cross={d_cross}; d={d_model}; el={e_n_layers}; dl={d_n_layers}-symbol_type={symbol_type}-pos_enc={pos_enc_type}'
if not update_symbols_each_layer:
    group_name += '-fxsym'
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

train_ds = torch.load(f'{data_path}/{task}_{args.train_postfix}.pt')
interp_ds = torch.load(f'{data_path}/{task}_interpolate.pt')
extrap_ds = torch.load(f'{data_path}/{task}_interpolate.pt')

# create data loaders
shuffle_train_dl = bool(args.shuffle)
num_workers = 4
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train_dl, num_workers=num_workers, drop_last=True, pin_memory=True)
interp_dl = torch.utils.data.DataLoader(interp_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=True)
extrap_dl = torch.utils.data.DataLoader(extrap_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=True)
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

        tf_acc = compute_tf_acc(logits, z, ignore_index=empty_token)

        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_teacher_forcing_acc', tf_acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate, betas=(0.9, 0.995)) # lr = 6e-4
        return optimizer
# endregion

# region build model
ra_kwargs = dict()
if symbol_type == 'symbolic_attention':
    symbol_retrieval_kwargs = dict(d_model=d_model, n_symbols=d_model, n_heads=4) # NOTE: n_heads, n_symbols fixed for now
elif symbol_type == 'position_relative':
    symbol_retrieval_kwargs = dict(symbol_dim=d_model, max_rel_pos=max_q_len)
    ra_kwargs['use_relative_positional_symbols'] = True # if using position-relative symbols, need to tell RA module
elif e_ra != 0 or d_ra!=0:
    raise ValueError(f'`symbol_type` {symbol_type} not valid')

# if ra=0, use TransformerLM
if e_ra == 0 and d_ra == 0:
    model_args = dict(
    input_spec=dict(type='token', vocab_size=vocab_size), output_spec=dict(type='token', vocab_size=vocab_size),
    pos_enc_type=pos_enc_type,
    d_model=d_model, out_dim=vocab_size, n_layers_enc=e_n_layers, n_layers_dec=d_n_layers,
    encoder_kwargs=dict(n_heads=e_sa, dff=dff, activation=activation, norm_first=False, dropout_rate=dropout_rate, causal=False),
    decoder_kwargs=dict(n_heads=d_sa, n_heads_cross=d_cross, dff=dff, activation=activation, norm_first=False, dropout_rate=dropout_rate, causal=True),
    in_block_size=max_q_len, out_block_size=max_a_len, loss_ignore_idx=0)
    model = Seq2SeqTransformer(**model_args)#.to(device)

# otherwise, use DualAttnTransformerLM
else:
    model_args = dict(
        input_spec=dict(type='token', vocab_size=vocab_size), output_spec=dict(type='token', vocab_size=vocab_size),
        pos_enc_type=pos_enc_type, update_symbols_each_layer=update_symbols_each_layer,
        symbol_retrieval=symbol_type, symbol_retrieval_kwargs=symbol_retrieval_kwargs,
        d_model=d_model, out_dim=vocab_size, n_layers_enc=e_n_layers, n_layers_dec=d_n_layers,
        encoder_kwargs=dict(n_heads_sa=e_sa, n_heads_ra=e_ra, ra_type=ra_type, ra_kwargs=ra_kwargs,
            dff=dff, activation=activation, norm_first=False, dropout_rate=dropout_rate, causal=False),
        decoder_kwargs=dict(n_heads_sa=d_sa, n_heads_ra=d_ra, n_heads_cross=d_cross, ra_type=ra_type, ra_kwargs=ra_kwargs,
            dff=dff, activation=activation, norm_first=False, dropout_rate=dropout_rate, causal=True),
        in_block_size=max_q_len, out_block_size=max_a_len, loss_ignore_idx=0)
    model = Seq2SeqDualAttnTransformer(**model_args)#.to(device)

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
    max_epochs=n_epochs, enable_checkpointing=False, enable_model_summary=True, precision='bf16-mixed',
    callbacks=callbacks, logger=wandb_logger, enable_progress_bar=True, check_val_every_n_epoch=5,
    )
trainer.fit(model=lit_model, train_dataloaders=train_dl, val_dataloaders=[interp_dl, extrap_dl])

wandb.finish()
# endregion