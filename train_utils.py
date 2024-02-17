import torch
import os
import time
from contextlib import nullcontext


def train_model(
    model, get_batch, batch_size, max_iters, optimizer, scaler, get_lr, eval_model,
    compile=True, grad_clip=0, gradient_accumulation_steps=1,
    eval_main_metric='val/loss', eval_interval=250,
    always_save_checkpoint=False, ckpt_dict=None, out_dir='out',
    log_interval=10, wandb_log=False, wandb_init_kwargs=None, track_mfu=True,
    master_process=True, ddp=False, device_type='cuda'):

    # set up wandb
    if wandb_log:
        import wandb
        wandb.init(**wandb_init_kwargs)

    # compilemodel
    if compile:
        print('compiling model...', end=' ')
        unoptimized_model = model
        model = torch.compile(model)
        print('done compiling.')

    if ckpt_dict is None:
        ckpt_dict = dict()

    # initialize tranining loop
    iter_num = 0
    best_val_loss = 1e9

    # TODO: infer batch_size for get_batch
    batch = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = None

    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    print('starting training loop...')
    # training loop
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the model on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            eval_metrics = eval_model(model, ctx)
            print(f"step {iter_num}: train loss {eval_metrics['train/loss']:.4f}, val loss {eval_metrics['val/loss']:.4f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    **eval_metrics,
                    "lr": lr,
                    "mfu": running_mfu, # convert to percentage
                })
            if eval_metrics[eval_main_metric] < best_val_loss or always_save_checkpoint:
                best_val_loss = eval_metrics[eval_main_metric]
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        **ckpt_dict # includes model_args, config, etc
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        # FIXME: make ddp work and test it
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(*batch)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            batch = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt) # FIXME: make tracking mfu optional
                running_mfu = mfu if running_mfu is None else 0.9*running_mfu + 0.1*mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break

    if wandb_log:
        wandb.finish()
    print('DONE.')