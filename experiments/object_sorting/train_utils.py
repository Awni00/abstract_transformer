import torch
import os
import time
from contextlib import nullcontext

def train_model(
    model, train_dl, eval_model, n_epochs,
    optimizer, scaler, get_lr,
    compile=True, grad_clip=0,
    eval_main_metric='val/loss',
    always_save_checkpoint=False, ckpt_dict=None, out_dir='out',
    wandb_log=False, wandb_init_kwargs=None, track_mfu=True,
    ddp=False, device_type='cuda'):

    # set up wandb
    if wandb_log:
        import wandb
        wandb.init(**wandb_init_kwargs)

    # compile model
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

    batch_size = train_dl.batch_size
    t0 = time.time()
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = None

    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    print('starting training loop...')
    # training loop
    for epoch in range(n_epochs):

        for batch in train_dl:

            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            with ctx:
                logits, loss = model(*batch)

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

            iter_num += 1

        # evaluate model, log to console, and log to W&B every epoch
        eval_metrics = eval_model(model, ctx)

        print(f"epoch: {epoch}, step: {iter_num}", end=" ")
        for k, v in eval_metrics.items():
            print(f"{k}: {v:.4f}", end=", ")
        print("🤖")

        if wandb_log:
            wandb.log({
                "epoch": epoch,
                "iter": iter_num,
                **eval_metrics,
                "lr": lr,
                "mfu": running_mfu,
            })
        if eval_metrics[eval_main_metric] < best_val_loss or always_save_checkpoint:
            best_val_loss = eval_metrics[eval_main_metric]
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                **ckpt_dict # includes model_args, config, etc
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        lossf = loss.item()
        mfu = raw_model.estimate_mfu(batch_size*len(train_dl), dt) # FIXME: make tracking mfu optional
        running_mfu = mfu if running_mfu is None else 0.9*running_mfu + 0.1*mfu
        print(f"epoch {epoch}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    if wandb_log:
        wandb.finish()
    print('DONE.')