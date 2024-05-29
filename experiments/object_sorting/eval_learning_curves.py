import argparse
import os
import numpy as np
import torch
import wandb
# import torchinfo
from contextlib import nullcontext
from tqdm import tqdm, trange

import sys; sys.path += ['../', '../..']
import train_utils
from seq2seq_models import Seq2SeqDualAttnTransformer, Seq2SeqTransformer
# from lightning_utils import LitSeq2SeqModel
import lightning as L

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# parse script arguments
parser = argparse.ArgumentParser()

parser.add_argument('--ee', type=int, help='number of encoder self-attention heads')
parser.add_argument('--ea', type=int, help='number of encoder relational cross-attention heads')
parser.add_argument('--de', type=int, help='number of decoder self-attention heads')
parser.add_argument('--da', type=int, help='number of decoder relational cross-attention heads')
parser.add_argument('--e_n_layers', type=int, help='number of encoder layers')
parser.add_argument('--d_n_layers', type=int, help='number of decoder layers')

# parser.add_argument('--eval_task_data_path', default='object_sorting_datasets/task2_object_sort_dataset.npy',
    # type=str, help='path to npy file containing sorting task dataset')
parser.add_argument('--n_epochs', default=2500, type=int, help='number of epochs to train each model for')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
# parser.add_argument('--min_train_size', default=500, type=int, help='minimum training set size')
# parser.add_argument('--max_train_size', default=5000, type=int, help='maximum training set size')
# parser.add_argument('--train_size_step', default=50, type=int, help='training set step size')
parser.add_argument('--num_trials', default=1, type=int, help='number of trials per training set size')
parser.add_argument('--start_trial', default=0, type=int, help='what to call first trial')
parser.add_argument('--wandb_project_name', default='abstract_transformer--object_sorting',
    type=str, help='W&B project name')
args = parser.parse_args()

batch_size = args.batch_size
num_trials = args.num_trials
start_trial = args.start_trial
n_epochs = args.n_epochs
wandb_project_name = args.wandb_project_name
train_sizes = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]

ee, ea, de, da = args.ee, args.ea, args.de, args.da
e_n_layers = args.e_n_layers
d_n_layers = args.d_n_layers

group_name = f'ee={ee}; ea={ea}; de={de}; da={da}; el={e_n_layers}; dl={d_n_layers}'

# region some configuration
device = 'cuda'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
# dtype = 'float32'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# wandb logging
wandb_log = False
wandb_project = 'abstract_transformer--object_sorting'
# endregion

# region data setup
data_path = 'object_sorting_datasets/task1_object_sort_dataset.npy'
data = np.load(data_path, allow_pickle=True).item()

objects, seqs, sorted_seqs, object_seqs, target, labels, start_token = tuple(
    data[key] for key in ['objects', 'seqs', 'sorted_seqs', 'object_seqs', 'target', 'labels', 'start_token'])

# convert to torch tensors
object_seqs = torch.tensor(object_seqs, dtype=ptdtype, device=device)
target = torch.tensor(target, dtype=torch.long, device=device)
labels = torch.tensor(labels, dtype=torch.long, device=device)
def train_val_test_split(*arrays, val_size=0.1, test_size=0.2):
    n = len(arrays[0])
    indices = np.random.permutation(n)
    val_start = int(n * (1 - val_size - test_size))
    test_start = int(n * (1 - test_size))
    train_indices = indices[:val_start]
    val_indices = indices[val_start:test_start]
    test_indices = indices[test_start:]
    return tuple(tuple(array[idx] for idx in (train_indices, val_indices, test_indices)) for array in arrays)
(object_seqs_train, object_seqs_val, object_seqs_test), (target_train, target_val, target_test), (labels_train, labels_val, labels_test) = train_val_test_split(
    object_seqs, target, labels, val_size=0.1, test_size=0.2)
print(f'training shapes: {object_seqs_train.shape}, {target_train.shape}, {labels_train.shape}')
print(f'validation shapes: {object_seqs_val.shape}, {target_val.shape}, {labels_val.shape}')
print(f'test shapes: {object_seqs_test.shape}, {target_test.shape}, {labels_test.shape}')

val_size = 512
val_ds = torch.utils.data.TensorDataset(object_seqs_val[:val_size], target_val[:val_size], labels_val[:val_size])
test_ds = torch.utils.data.TensorDataset(object_seqs_test, target_test, labels_test)

val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

def get_train_dl(train_size, batch_size=batch_size):
    sample_idx = np.random.choice(object_seqs_train.shape[0], train_size)
    train_ds = torch.utils.data.TensorDataset(object_seqs_train[sample_idx], target_train[sample_idx], labels_train[sample_idx])
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

    return train_dl

# endregion

# region Lightning Setup
class LitSeq2SeqModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y, z = batch
        # with ctx:
        logits, loss = self.model(x, y, z)

        self.log('loss', loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, z = batch
        # with ctx:
        logits, loss = self.model(x, y, z)
        tf_acc = torch.mean((torch.argmax(logits, dim=-1) == z).float())
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_tf_acc", tf_acc, prog_bar=True, logger=True)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# endregion


# region model creation setup
model_args = dict(
    input_spec=dict(type='vector', dim=8), output_spec=dict(type='token', vocab_size=10+1),
    symbol_retrieval='pos_sym_retriever', symbol_retrieval_kwargs=dict(symbol_dim=64, max_symbols=10),
    d_model=64, out_dim=10, n_layers_enc=e_n_layers, n_layers_dec=d_n_layers,
    encoder_kwargs=dict(n_heads_enc=ee, n_heads_abs=ea, dff=64, activation='relu', norm_first=False, dropout_rate=0.1, causal=False, rel_mask_diag=False),
    decoder_kwargs=dict(n_heads_enc=de, n_heads_abs=da, n_heads_cross=2, dff=64, activation='relu', norm_first=False, dropout_rate=0.1, causal=True, rel_mask_diag=False),
    in_block_size=10, out_block_size=10)

def create_model():
    return Seq2SeqDualAttnTransformer(**model_args)

# endregion

# region learning curve evaluation utilities
def evaluate_learning_curves(
    create_model,
    wandb_project_name, group_name,
    train_sizes=train_sizes, num_trials=num_trials):

    for train_size in tqdm(train_sizes, desc='train size'):

        for trial in trange(start_trial, start_trial + num_trials, desc='trial', leave=False):
            run = wandb.init(project=wandb_project_name, group=group_name, name=f'train size = {train_size}; trial = {trial}',
                            config={'train size': train_size, 'trial': trial, 'group': group_name, **model_args})

            model = create_model()
            lit_model = LitSeq2SeqModel(model)

            train_dl = get_train_dl(train_size)

            callbacks = [
                # EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=True)
                ]

            trainer = L.Trainer(
                max_epochs=n_epochs, enable_checkpointing=False, enable_model_summary=True, precision='64-true', callbacks=callbacks,
                enable_progress_bar=True,#check_val_every_n_epoch=50, logger=logger,
                )
            trainer.fit(model=lit_model, train_dataloaders=train_dl) # , val_dataloaders=val_dl)

            lit_model.eval()

            eval_dict = trainer.test(lit_model, test_dl)[0]
            wandb.log(eval_dict)

            wandb.finish(quiet=False)

            del model

# endregion


# region learning curve evaluation

evaluate_learning_curves(
    create_model, wandb_project_name, group_name,
    train_sizes=train_sizes, num_trials=num_trials)
# endregion