# Sample Efficient Relational Reasoning: Relational Games

This set of experiments aims to evaluate the effect of the dual attention mechanism on sample efficiency in learning relational reasoning. We use the "Relational Games" benchmark contributed by [Shanahan et al. (2020)](https://arxiv.org/abs/1905.10307) which consists of a suite of binary classification tasks based on identifying visual relationships among objects in an image.

## Instructions to replicate experimental results

To replicate the experimental results reported in the paper, please follow the instructions below.

**Prepare data**

Download the relational games dataset. It is hosted on google cloud platform [here](https://console.cloud.google.com/storage/browser/relations-game-datasets) and linked through [Shanahan et al.'s github repo](https://github.com/google-deepmind/deepmind-research/tree/master/PrediNet). The dataset files are hosted in npz format. Place the data under a directory named `data/relational_games/npz_files` in the project root.

Next, extract the data and convert to pytorch format by running `process_npz_data.py`. This will create `.pt` files for each data file and place it in `data/relational_games`.

**Evaluate Learning Curves**

Learning curves can be evaluated for each model configuration with the `eval_relational_games_learning_curve.py` script. The script receives a specification of the task, model configuration, etc., and evaluates learning curves. The script logs the results to the *Weights & Biases* experiment tracking platform. Run `python eval_relational_games_learning_curve.py --help` for an explanation of each argument.

```
usage: eval_relational_games_learning_curve.py [-h] --task TASK --sa SA --ra RA --symbol_type {positional_symbols,position_relative,symbolic_attention,NA} --ra_type
                                               {relational_attention,rca,disrca,NA} --n_layers N_LAYERS --d_model D_MODEL --dff DFF [--activation ACTIVATION] [--dropout_rate DROPOUT_RATE]
                                               [--norm_first NORM_FIRST] [--symmetric_rels SYMMETRIC_RELS] [--patch_size PATCH_SIZE] [--pool POOL] --train_sizes TRAIN_SIZES [TRAIN_SIZES ...]
                                               [--n_trials N_TRIALS] [--val_size VAL_SIZE] [--test_size TEST_SIZE] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
                                               [--wandb_project WANDB_PROJECT] [--early_stopping EARLY_STOPPING] [--eval_interval EVAL_INTERVAL] [--log_every_n_steps LOG_EVERY_N_STEPS]
                                               [--max_steps MAX_STEPS] [--log_model LOG_MODEL] [--log_to_wandb LOG_TO_WANDB] [--compile COMPILE]

options:
  -h, --help            show this help message and exit
  --task TASK           relational games task
  --sa SA               number of self-attention heads
  --ra RA               number of relational attention heads
  --symbol_type {positional_symbols,position_relative,symbolic_attention,NA}
                        type of symbols to use
  --ra_type {relational_attention,rca,disrca,NA}
                        type of relational attn to use
  --n_layers N_LAYERS   number of layers
  --d_model D_MODEL     model dimension
  --dff DFF             feedforward hidden dimension
  --activation ACTIVATION
                        MLP activation
  --dropout_rate DROPOUT_RATE
                        dropout rate
  --norm_first NORM_FIRST
                        whether to use pre-LN or post-LN
  --symmetric_rels SYMMETRIC_RELS
                        whether to impose symmetric relations in RA
  --patch_size PATCH_SIZE
                        size of patches for ViT
  --pool POOL           type of pooling operation to use
  --train_sizes TRAIN_SIZES [TRAIN_SIZES ...]
                        training set sizes for learning curves
  --n_trials N_TRIALS   training set sizes for learning curves
  --val_size VAL_SIZE   validation set size
  --test_size TEST_SIZE
                        test set size
  --n_epochs N_EPOCHS   number of passes through data to train for
  --batch_size BATCH_SIZE
                        batch size
  --learning_rate LEARNING_RATE
                        learning rate
  --wandb_project WANDB_PROJECT
                        W&B project name
  --early_stopping EARLY_STOPPING
                        whether to use early stopping
  --eval_interval EVAL_INTERVAL
                        interval of evaluating validation set
  --log_every_n_steps LOG_EVERY_N_STEPS
                        interval of logging training metrics
  --max_steps MAX_STEPS
                        maximum number of steps
  --log_model LOG_MODEL
                        whether to save the model at the end of training
  --log_to_wandb LOG_TO_WANDB
                        whether to log to wandb
  --compile COMPILE     whether to compile model
```

To replicate the results reported in the paper, run this script with the configuration detailed in the "Experimental Details & Further Discussion" section of the paper's appendix, or use the command for each run in the experimental logs.

## Experimental logs

For each run, the experimental logs include: the git commit ID giving the version of the code that was used to run the experiment, the exact command and arguments used to run the script, the hardware used for that run, and evaluated metrics for each model configuration. Experimental logs can be found at the: [W&B project link](https://wandb.ai/awni00/dual_attention--relational_games_learning_curves?nw=nwuserawni00)