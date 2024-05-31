# Vision: Image Recognition

This set of experiments aims to evaluate the effect of the dual attention mechanism on vision tasks. We use the [ImageNet](https://image-net.org/) Large Scale Visual Recognition Challenge 2012 (ILSVRC2012) dataset for image recognition. This experiment uses a "Vision Transformer" style architecture, where the images are split up into patches, embedded, then passed through a standard Transformer.

## Instructions to replicate experimental results

To replicate the experimental results reported in the paper, please follow the instructions below.

**Prepare data**

Download the Imagenet ILSVRC2012 dataset. There are several ways to obtain this data. A convenient way is through [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data?source=post_page-----ac90f3db5cf9--------------------------------&select=ILSVRC) by running `kaggle competitions download -c imagenet-object-localization-challenge`. The `imagenet_data_utils.py` file defines a PyTorch Dataset interface for reading this data.

**Evaluate Model**

Each model configuration can be evaluated with the `pretrain_imagenet_vision_model.py` script. The script receives a specification of the model configuration and trains/evaluates the language model. The script logs the results to the *Weights & Biases* experiment tracking platform. Run `python pretrain_imagenet_vision_model.py --help` for an explanation of each argument.

```
usage: pretrain_imagenet_vision_model.py [-h] --sa SA --ra RA --symbol_type {positional_symbols,position_relative,symbolic_attention,NA} --ra_type {relational_attention,rca,disrca,NA} --n_layers
                                         N_LAYERS --d_model D_MODEL [--activation ACTIVATION] [--dropout_rate DROPOUT_RATE] [--norm_first NORM_FIRST] [--symmetric_rels SYMMETRIC_RELS] [--dff DFF]
                                         [--patch_size PATCH_SIZE] [--pool POOL] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
                                         [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--wandb_project WANDB_PROJECT] [--eval_interval EVAL_INTERVAL]
                                         [--log_every_n_steps LOG_EVERY_N_STEPS] [--max_steps MAX_STEPS] [--log_model LOG_MODEL] [--log_to_wandb LOG_TO_WANDB] [--compile COMPILE] [--resume RESUME]
                                         [--ckpt_path CKPT_PATH] [--run_id RUN_ID]

options:
  -h, --help            show this help message and exit
  --sa SA               number of self-attention heads
  --ra RA               number of relational attention heads
  --symbol_type {positional_symbols,position_relative,symbolic_attention,NA}
                        type of symbols to use
  --ra_type {relational_attention,rca,disrca,NA}
                        type of RA to use
  --n_layers N_LAYERS   number of layers
  --d_model D_MODEL     model dimension
  --activation ACTIVATION
                        MLP activation
  --dropout_rate DROPOUT_RATE
                        dropout rate
  --norm_first NORM_FIRST
                        whether to use pre-LN or post-LN
  --symmetric_rels SYMMETRIC_RELS
                        whether to impose symmetric relations in RA
  --dff DFF             feedforward hidden dimension
  --patch_size PATCH_SIZE
                        size of patches for ViT
  --pool POOL           type of pooling operation to use
  --n_epochs N_EPOCHS   number of passes through data to train for
  --batch_size BATCH_SIZE
                        batch size
  --learning_rate LEARNING_RATE
                        learning_rate
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        gradient_accumulation_steps
  --wandb_project WANDB_PROJECT
                        W&B project name
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
  --compile COMPILE     whether to compile
  --resume RESUME       whether to resume from a previous run
  --ckpt_path CKPT_PATH
                        path to checkpoint
  --run_id RUN_ID       W&B run ID for resuming
```

To replicate the results reported in the paper, run this script with the configuration detailed in the "Experimental Details & Further Discussion" section of the paper's appendix, or use the command for each run in the experimental logs.

## Experimental logs

For each run, the experimental logs include: the git commit ID giving the version of the code that was used to run the experiment, the exact command and arguments used to run the script, the hardware used for that run, and evaluated metrics for each model configuration. Experimental logs can be found at the: [W&B project link](https://wandb.ai/awni00/dual_attention--Vision-IMAGENET)