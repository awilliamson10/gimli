"""
This training script can be run both on a single gpu in debug mode, and also in a larger training run with distributed data parallel (ddp).

- You can run this script on a single gpu with the following command:
    python gimli/train.py --config=configs/train_config.yaml
"""
from tqdm import tqdm
import accelerate
import os
import torch
from config import global_config as config
from gimli.export import model_export
from functools import partial
from gimli.dataloader import Task
from gimli.model import Transformer, ModelArgs
from gimli.scheduler import cosine_lr

accelerator = accelerate.Accelerator(
    mixed_precision=config.dtype if config.dtype != "float32" else "no",
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with="wandb" if config.wandb_log else None,
    dispatch_batches=True, # This is used when using IterableDataset
)


def init_process_group():
    raise NotImplementedError


def init_model(init_from, model_args):
    if init_from == "scratch":
        accelerator.print(f"Initializing model from scratch")
        model_args = ModelArgs(**model_args)
        model = Transformer(model_args)
    elif init_from == "checkpoint":
        raise NotImplementedError(f"Checkpoint init not implemented yet")
    else:
        raise ValueError(f"Unknown init_from: {init_from}")
    if config.compile:
        accelerator.print("Compiling model (this could take a while)...")
        model = torch.compile(model)
    return model


def save_checkpoint(out_dir, model, optimizer, model_args, iter_num, best_val_loss, config):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    model_export(model, os.path.join(out_dir, "model.bin"), version=0)


def train(model, optimizer, scheduler, iter_batches, model_args):
    train_batch_iter = iter_batches(split="train")
    val_batch_iter = iter_batches(split="val")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(config.max_iters),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    global_step = 0
    last_save = 0

    while True:
        if global_step % config.eval_interval == 0 and accelerator.is_main_process:
            model.eval()
            val_loss = 0.0
            for _ in range(config.eval_iters):
                with torch.no_grad():
                    X, Y = next(val_batch_iter)
                    _, loss = model(X, Y)
                    val_loss += loss.item()
            val_loss /= config.eval_iters
            accelerator.print(f"Validation loss: {val_loss:.4f}")
            accelerator.save()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if accelerator.is_main_process:
                    save_checkpoint(
                        config.out_dir,
                        model,
                        optimizer,
                        model_args,
                        global_step,
                        best_val_loss,
                        config,
                    )
                    last_save = global_step
        if global_step == 0 and config.eval_only:
            break

        # forward backward update, with gradient accumulation
        model.train()
        for micro_step in range(config.gradient_accumulation_steps):
            X, Y = next(train_batch_iter)
            _, loss = model(X, Y)
            loss = loss / config.gradient_accumulation_steps
            accelerator.backward(loss)
        if config.grad_clip > 0:
            accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler(global_step)
        optimizer.zero_grad(set_to_none=True)

        # update progress bar
        progress_bar.update(1)
        global_step += 1
        logs = {
            "loss": loss.item(),
            "lr": optimizer.param_groups[0]["lr"],
            "step": global_step,
        }
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if global_step >= config.max_iters:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.print(f"Saving final model to {config.out_dir}")
        save_checkpoint(
            config.out_dir,
            model,
            optimizer,
            model_args,
            global_step,
            best_val_loss,
            config,
        )
    

def main():
    # validating checks
    assert config.vocab_source in ["llama2", "custom"]
    assert config.vocab_source == "custom" or config.vocab_size == 32000, "The vocab from Meta has 32K tokens"

    if accelerator.is_main_process:
        tokens_per_iter = config.gradient_accumulation_steps * config.batch_size * config.max_seq_len # * ddp_world_size 
        accelerator.print(f"tokens per iteration will be: {tokens_per_iter:,}")
        accelerator.print(f"breaks down as: {config.gradient_accumulation_steps} grad accum steps * {config.batch_size} batch size * {config.max_seq_len} max seq len")

        # create output directory
        accelerator.print(f"Creating output directory: {config.out_dir}")
        os.makedirs(config.out_dir, exist_ok=True)

        if config.wandb_log:
            accelerator.init_trackers(
                project_name=config.wandb_project,
                init_kwargs={"run_name": config.wandb_run_name},
                config=config,
            )
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    seed_offset = 0
    if ddp:
        init_process_group()
        seed_offset = accelerator.rank

    # reproducibility
    torch.manual_seed(42 + seed_offset)

    # task-specific setup
    iter_batches = partial(
        Task.iter_batches,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        vocab_size=config.vocab_size,
        vocab_source=config.vocab_source,
        device=config.device,
        num_workers=0,
    )

    # model init
    model_args = dict(
        dim=config.dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        vocab_size=config.vocab_size,
        multiple_of=config.multiple_of,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    )
    model = init_model(model_args)

    # optimizer
    optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), config.device_type)
    scheduler = cosine_lr(optimizer, config.learning_rate, config.warmup_iters, config.max_iters)

    # wrap model and optimizer with accelerator
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    model.to(accelerator.device)

    # training loop
    train(model, optimizer, scheduler, iter_batches, model_args)

    accelerator.print("Training finished!")
    accelerator.end_training()


if __name__ == "__main__":
    main()