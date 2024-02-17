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
from seq2seq_models import Seq2SeqAbstractTransformer, Seq2SeqTransformer

# parse script arguments
parser = argparse.ArgumentParser()

parser.add_argument('--ee', type=int, help='number of encoder self-attention heads')
parser.add_argument('--ea', type=int, help='number of encoder relational cross-attention heads')
parser.add_argument('--de', type=int, help='number of decoder self-attention heads')
parser.add_argument('--da', type=int, help='number of decoder relational cross-attention heads')

# parser.add_argument('--eval_task_data_path', default='object_sorting_datasets/task2_object_sort_dataset.npy',
    # type=str, help='path to npy file containing sorting task dataset')
parser.add_argument('--n_epochs', default=500, type=int, help='number of epochs to train each model for')
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
train_sizes = [250, 500, 1000, 1500, 2000, 2500, 3000]

ee, ea, de, da = args.ee, args.ea, args.de, args.da

group_name = f'ee={ee}; ea={ea}; de={de}; da={da}'

# region some configuration
# I/O
eval_only = False # if True, script exits right after the first eval

# system
# device = 'cpu'
device = 'cuda'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

# 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# dtype = 'float32'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
compile = True

# evaluation and output
out_dir = '../out/object_sorting'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

# wandb logging
wandb_log = False
wandb_project = 'abstract_transformer--object_sorting'

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
warmup_iters = 100
gradient_accumulation_steps = 1 # accumulate gradients over this many steps. simulates larger batch size

# DDP (distributed data parallel) training
ddp = False
master_process = True
# TODO: set up DDP for future experiments

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
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
train_size = 1500
sample_idx = np.random.choice(object_seqs_train.shape[0], train_size)

train_ds = torch.utils.data.TensorDataset(object_seqs_train[sample_idx], target_train[sample_idx], labels_train[sample_idx])
val_ds = torch.utils.data.TensorDataset(object_seqs_val, target_val, labels_val)
test_ds = torch.utils.data.TensorDataset(object_seqs_test, target_test, labels_test)


train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
for source, target, label in train_dl:
    print(source.shape, target.shape, label.shape)
    break

def get_train_dl(train_size, batch_size=batch_size):
    sample_idx = np.random.choice(object_seqs_train.shape[0], train_size)
    train_ds = torch.utils.data.TensorDataset(object_seqs_train[sample_idx], target_train[sample_idx], labels_train[sample_idx])
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

    return train_dl

# endregion

# region training setup
def get_lr(it):
    return 0.001

@torch.no_grad()
def eval_model(model, ctx=None):

    ctx = nullcontext() if ctx is None else ctx
    out = {}
    model.eval()
    for split in ['train', 'val']:
        dl = train_dl if split == 'train' else val_dl
        max_batches = min(eval_iters, len(dl)) if eval_iters is not None else len(dl)
        losses = torch.zeros(max_batches)
        tfaccs = torch.zeros(max_batches)
        for k, batch in enumerate(dl):
            source, target, label = batch
            if eval_iters is not None and k >= max_batches:
                break
            with ctx:
                logits, loss = model(source, target, label)
            losses[k] = loss.item()
            tfaccs[k] = torch.mean((torch.argmax(logits, dim=-1) == label).float())

        out[f'{split}/loss'] = losses.mean() # FIXME loss is averaged over batch. batch sizes may be unnequal?
        out[f'{split}/tfacc'] = tfaccs.mean()
    model.train()
    return out

@torch.no_grad()
def evaluate_seq2seq_model(model, source_test, target_test, labels_test, start_token, print_=False, ctx=ctx):

    model.eval()

    n, seqs_length = target_test.shape
    output = torch.zeros(size=(n, (seqs_length+1)), dtype=torch.int, device=device)
    output[:,0] = start_token

    for i in range(seqs_length):
        with ctx:
            predictions, _ = model(source_test, output[:, :-1], labels_test)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)
        output[:,i+1] = predicted_id

    elementwise_acc = torch.mean((output[:,1:] == labels_test).float()).item()
    acc_per_position = [torch.mean((output[:, i+1] == labels_test[:, i]).float()).item() for i in range(seqs_length)]
    seq_acc = torch.mean((torch.all(output[:,1:]==labels_test, axis=1)).float()).item()

    with ctx:
        tf_pred = model(source_test, target_test, labels_test)[0]
        tf_pred = torch.argmax(tf_pred, axis=-1)
    teacher_forcing_acc = torch.mean((labels_test==tf_pred).float()).item()

    if print_:
        print('element-wise accuracy: %.2f%%' % (100*elementwise_acc))
        print('full sequence accuracy: %.2f%%' % (100*seq_acc))
        print('teacher-forcing accuracy:  %.2f%%' % (100*teacher_forcing_acc))


    return_dict = {
        'elementwise_accuracy': elementwise_acc, 'full_sequence_accuracy': seq_acc,
        'teacher_forcing_accuracy': teacher_forcing_acc, 'acc_by_position': acc_per_position
        }

    return return_dict
# endregion

# region model creation setup
def create_abstransformer_model(ee, ea, de, da):

    model_args = dict(
        input_spec=dict(type='vector', dim=8), output_spec=dict(type='token', vocab_size=10+1),
        symbol_retrieval='pos_sym_retriever', symbol_retrieval_kwargs=dict(symbol_dim=64, max_symbols=10),
        d_model=64, out_dim=10, n_layers_enc=2, n_layers_dec=2,
        encoder_kwargs=dict(n_heads_enc=ee, n_heads_abs=ea, dff=128, activation='relu', norm_first=True, dropout_rate=0.1, causal=False),
        decoder_kwargs=dict(n_heads_enc=de, n_heads_abs=da, n_heads_cross=2, dff=128, activation='relu', norm_first=True, dropout_rate=0.1, causal=True),
        in_block_size=10, out_block_size=10)
    seq2seqabstransformer = Seq2SeqAbstractTransformer(**model_args)#.to(device)
    return seq2seqabstransformer

def create_model():
    return create_abstransformer_model(ee, ea, de, da)

# endregion

# region learning curve evaluation utilities
def evaluate_learning_curves(
    create_model,
    wandb_project_name, group_name,
    train_sizes=train_sizes, num_trials=num_trials):

    for train_size in tqdm(train_sizes, desc='train size'):

        for trial in trange(start_trial, start_trial + num_trials, desc='trial', leave=False):
            # run = wandb.init(project=wandb_project_name, group=group_name, name=f'train size = {train_size}; trial = {trial}',
                            # config={'train size': train_size, 'trial': trial, 'group': group_name})
            # TODO: add model args to config?

            model = create_model().to(device)

            scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
            # optimizer
            optimizer = torch.optim.Adam(model.parameters())
            # optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type=device)

            train_dl = get_train_dl(train_size)

            # TODO: make training loop support pre-initiated wanbd runs
            train_kwargs = dict(
                model=model, train_dl=train_dl, eval_model=eval_model, n_epochs=n_epochs,
                optimizer=optimizer, scaler=scaler, get_lr=get_lr,
                compile=True, grad_clip=0,
                eval_main_metric='val/loss',
                always_save_checkpoint=always_save_checkpoint,
                # ckpt_dict=dict(model_args=model_args), 
                out_dir=out_dir,
                wandb_log=False, wandb_init_kwargs=dict(project=wandb_project, group=group_name, name=f'{group_name}--trial={trial}'),
                track_mfu=True,
                ddp=False, device_type='cuda')
            train_utils.train_model(**train_kwargs)

            source_test, target_test, labels_test = test_ds.tensors
            eval_dict = evaluate_seq2seq_model(model, source_test, target_test, labels_test, start_token, print_=True, ctx=ctx)

            # wandb.log(eval_dict)
            # wandb.finish(quiet=True)

            del model

# endregion


# region learning curve evaluation

evaluate_learning_curves(
    create_model, wandb_project_name, group_name,
    train_sizes=train_sizes, num_trials=num_trials)
# endregion