# Language Modeling

This set of experiments aims to evaluate the effect of the dual attention mechanism on autoregressive language modeling. We use the "Tiny Stories" benchmark contributed by [Eldan and Li (2023)](https://arxiv.org/abs/2305.07759) for evaluating small language models, which consists of a collections of short stories.

## Instructions to replicate experimental results

To replicate the experimental results reported in the paper, please follow the instructions below.

**Prepare data**

Download the tiny stories dataset by running `python tiny_stories_data.py download`. This will download `TinyStories_all_data.tar.gz` from [huggingface](https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz) and unpack it all data shards into `.json` files into the `experiments/tiny_stories/TinyStories_all_data` directory.

Tokenize the data (using the Llama SentencePiece tokenizer in `tokenizer.model`) by running `tiny_stories_data.py pretokenize`. The tokenized data shards are written to `.bin` files in `experiments/tiny_stories/TinyStories_all_data`.


**Evaluate Modells**

Each model configuration can be evaluated with the `train.py` script. The script receives a specification of the model configuration and trains/evaluates the language model. The script logs the results to the *Weights & Biases* experiment tracking platform. Run `python train.py --help` for an explanation of each argument.

```
usage: train.py [-h] --sa SA --ra RA --symbol_type {position_relative,symbolic_attention,NA} --pos_enc_type {RoPE,pos_emb} --ra_type {relational_attention,rca,disrca,NA} --n_layers N_LAYERS --d_model
                D_MODEL [--activation ACTIVATION] [--dropout_rate DROPOUT_RATE] [--dff DFF] [--norm_first NORM_FIRST] [--symmetric_rels SYMMETRIC_RELS] [--trainable_symbols TRAINABLE_SYMBOLS]
                [--max_seq_len MAX_SEQ_LEN] [--n_epochs N_EPOCHS] [--max_iters MAX_ITERS] [--batch_size BATCH_SIZE] [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                [--learning_rate LEARNING_RATE] [--use_cosine_sched USE_COSINE_SCHED] [--eval_interval EVAL_INTERVAL] [--eval_iters EVAL_ITERS] [--eval_only EVAL_ONLY] [--log_to_wandb LOG_TO_WANDB]
                [--always_save_checkpoint ALWAYS_SAVE_CHECKPOINT] [--compile COMPILE] [--init_from INIT_FROM] [--wandb_project WANDB_PROJECT]

options:
  -h, --help            show this help message and exit
  --sa SA               number of self-attention heads
  --ra RA               number of relational attention heads
  --symbol_type {position_relative,symbolic_attention,NA}
                        type of symbols to use
  --pos_enc_type {RoPE,pos_emb}
                        type of symbols to use
  --ra_type {relational_attention,rca,disrca,NA}
                        type of relational attn to use
  --n_layers N_LAYERS   number of layers
  --d_model D_MODEL     model dimension
  --activation ACTIVATION
                        MLP activation
  --dropout_rate DROPOUT_RATE
                        dropout rate
  --dff DFF             feedforward hidden dimension
  --norm_first NORM_FIRST
                        whether to use pre-norm or post-norm
  --symmetric_rels SYMMETRIC_RELS
                        whether to impose symmetric relations in RA
  --trainable_symbols TRAINABLE_SYMBOLS
                        whether to allow symbols to be trainable (for sym_attn only for now)
  --max_seq_len MAX_SEQ_LEN
                        max seq length / block size
  --n_epochs N_EPOCHS   number of passes through data to train for
  --max_iters MAX_ITERS
                        maximum number of steps
  --batch_size BATCH_SIZE
                        batch size
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        number of gradiient accumulation steps
  --learning_rate LEARNING_RATE
                        learning rate
  --use_cosine_sched USE_COSINE_SCHED
                        whether to use a cosine learning rate schedule
  --eval_interval EVAL_INTERVAL
                        interval of evaluating validation set
  --eval_iters EVAL_ITERS
                        # of iters to estimate val loss
  --eval_only EVAL_ONLY
                        whether to exit after first eval
  --log_to_wandb LOG_TO_WANDB
                        whether to log to wandb
  --always_save_checkpoint ALWAYS_SAVE_CHECKPOINT
                        whether to save ckpt after each eval
  --compile COMPILE     whether to compile
  --init_from INIT_FROM
                        whether to init from scratch or resume training
  --wandb_project WANDB_PROJECT
                        W&B project name
```

To replicate the results reported in the paper, run this script with the configuration detailed in the "Experimental Details & Further Discussion" section of the paper's appendix, or use the command for each run in the experimental logs.

## Experimental logs

For each run, the experimental logs include: the git commit ID giving the version of the code that was used to run the experiment, the exact command and arguments used to run the script, the hardware used for that run, and evaluated metrics for each model configuration. Experimental logs can be found at the: [W&B project link](https://wandb.ai/awni00/dual_attention--tiny_stories-LM)