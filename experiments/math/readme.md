# Improved Symbolic Reasoning in Sequence-to-Sequence tasks: Mathematical Problem Solving

This set of experiments aims to evaluate the effect of the dual attention mechanism on symbolic reasoning ability in sequence-to-sequence tasks. We use the mathematical problem-solving benchmark contributed by [Saxton et al. (2020)](https://arxiv.org/abs/1904.01557) which consists of a suite of mathematical prolem-solving tasks in free-form textual input/output format.

## Instructions to replicate experimental results

To replicate the experimental results reported in the paper, please follow the instructions below.

**Prepare data**

Download the dataset. It is hosted on google cloud platform [here](https://console.cloud.google.com/storage/browser/mathematics-dataset;tab=objects?prefix=&forceOnObjectsSortingFiltering=false) and linked through [Saxton et al.'s github repo](https://github.com/google-deepmind/mathematics_dataset?tab=readme-ov-file). The dataset is uploaded as a zipped `.tar.gz` file which contains the dataset split up across several modules and difficulty splits, each in `.txt` format. Unzip the file into a directory named `data/math/mathematics_dataset-v1.0` in the project root.

Next, process and tokenize the data by running `process_data.ipynb`. This will tokenize the data with a character-level encode, create `.pt` files for each data file, and place it in `experiments/math/tokenized data`.

**Evaluate Models**

Each model configuration can be evaluated with the `train_model.py` script. The script receives a specification of the task, model configuration, etc., and trains/evaluates the mode. The script logs the results to the *Weights & Biases* experiment tracking platform. Run `python train_model.py --help` for an explanation of each argument.

```
usage: train_model.py [-h] --task TASK --e_sa E_SA --e_ra E_RA --d_sa D_SA --d_ra D_RA --d_cross D_CROSS --symbol_type {position_relative,symbolic_attention,position_relative,NA} --ra_type
                      {relational_attention,rca,disrca,NA} --e_n_layers E_N_LAYERS --d_n_layers D_N_LAYERS --d_model D_MODEL [--activation ACTIVATION] [--dropout_rate DROPOUT_RATE] --dff DFF
                      [--learning_rate LEARNING_RATE] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE] [--shuffle SHUFFLE] [--train_postfix TRAIN_POSTFIX] [--run_name RUN_NAME]
                      [--wandb_project_prefix WANDB_PROJECT_PREFIX]

options:
  -h, --help            show this help message and exit
  --task TASK           task name
  --e_sa E_SA           number of encoder self-attention heads
  --e_ra E_RA           number of encoder relational attention heads
  --d_sa D_SA           number of decoder self-attention heads
  --d_ra D_RA           number of decoder relational attention heads
  --d_cross D_CROSS     number of decoder cross-attention heads
  --symbol_type {position_relative,symbolic_attention,position_relative,NA}
                        type of symbols to use
  --ra_type {relational_attention,rca,disrca,NA}
                        type of relational attn to use
  --e_n_layers E_N_LAYERS
                        number of encoder layers
  --d_n_layers D_N_LAYERS
                        number of decoder layers
  --d_model D_MODEL     model dimension
  --activation ACTIVATION
                        MLP activation
  --dropout_rate DROPOUT_RATE
                        dropout rate
  --dff DFF             feedforward hidden dimension
  --learning_rate LEARNING_RATE
                        learning rate
  --n_epochs N_EPOCHS   number of epochs to train each model for
  --batch_size BATCH_SIZE
                        batch size
  --shuffle SHUFFLE     whether to shuffle train loader (0 or 1)
  --train_postfix TRAIN_POSTFIX
                        training split (i.e,, train-easy, train-medium, train-hard, train)
  --run_name RUN_NAME   wandb run name
  --wandb_project_prefix WANDB_PROJECT_PREFIX
                        W&B project name
```

To replicate the results reported in the paper, run this script with the configuration detailed in the "Experimental Details & Further Discussion" section of the paper's appendix, or use the command for each run in the experimental logs.

## Experimental logs

For each run, the experimental logs include: the git commit ID giving the version of the code that was used to run the experiment, the exact command and arguments used to run the script, the hardware used for that run, and evaluated metrics for each model configuration. Experimental logs can be found at the links below.

| Task                        	| Experimental logs                                                                 	|
|-----------------------------	|-----------------------------------------------------------------------------------	|
| calculus__differentiate     	| https://wandb.ai/awni00/dual_attention--math--calculus__differentiate     	|
| algebra__sequence_next_term 	| https://wandb.ai/awni00/dual_attention--math--algebra__sequence_next_term 	|
| algebra__linear_1d          	| https://wandb.ai/awni00/dual_attention--math--algebra__linear_1            |
| polynomials__expand         	| https://wandb.ai/awni00/dual_attention--math--polynomials__expand         	|
| polynomials__add            	| https://wandb.ai/awni00/dual_attention--math--polynomials__add            	|