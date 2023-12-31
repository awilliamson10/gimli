"""
This training script can be run both on a single gpu in debug mode, and also in a larger training run with distributed data parallel (ddp).

- You can run this script on a single gpu with the following command:
    python gimli/train.py --config=configs/train_config.yaml
    
"""
import os
from contextlib import nullcontext
from functools import partial

import torch
import torch._dynamo
import wandb
from peft import LoraConfig, get_peft_model
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from gimli.config import global_config as config
from gimli.dataset import Task
from gimli.export import export_lora, load_checkpoint, model_export
from gimli.model import ModelArgs, Transformer, apply_lora
from gimli.scheduler import lr_scheduler

torch._dynamo.config.suppress_errors = True

def setup_ddp():
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group()
        ddp_rank = int(os.environ.get("RANK", -1))
        ddp_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        ddp_world_size = int(os.environ.get("WORLD_SIZE", -1))
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert (
            config.gradient_accumulation_steps % ddp_world_size == 0
        ), "gradient_accumulation_steps must be divisible by the number of ddp processes"
        config.gradient_accumulation_steps = (
            config.gradient_accumulation_steps // ddp_world_size
        )
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_local_rank = 0
    return ddp, master_process, seed_offset, ddp_world_size, ddp_local_rank


def init_model(init_from, model_args, config):
    if init_from == "scratch":
        print("Initializing model from scratch")
        model_args = ModelArgs(**model_args)
        model = Transformer(model_args)
    elif init_from in ["checkpoint", "lora"]:
        print("Initializing model from checkpoint")
        model = load_checkpoint(os.path.join(config.out_dir, "ckpt.pt"))
        if init_from == "lora":
            config.out_dir = os.path.join(config.out_dir, "lora")
            os.makedirs(config.out_dir, exist_ok=True)
            print(f"Preparing to train with LoRA, saving to {config.out_dir}")
            for name, param in model.named_parameters():
                param.requires_grad = False
            apply_lora(model, config.lora_rank, config.lora_dropout, config.lora_alpha, config.lora_target_modules)
            # make sure lora modules are trainable
            config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
            )
            model = get_peft_model(model, config)
        return model
    else:
        raise ValueError(f"Unknown init_from: {init_from}")
    if config.compile:
        print("Compiling model (this could take a while)...")
        model = torch.compile(model)
    return model


def save_checkpoint(
    out_dir, model, optimizer, model_args, iter_num, best_val_loss, config, lora=False
):
    if not lora:
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
    else:
        export_lora(model, os.path.join(out_dir, "model_lora.bin"), config.lora_target_modules)

def train(
    model,
    optimizer,
    scheduler,
    scaler,
    iter_batches,
    model_args,
    ctx,
    master_process=True,
    ddp=False,
):
    train_batch_iter = iter_batches(split="train")
    X, Y = next(train_batch_iter)

    global_step = 0
    best_val_loss = 1e9

    while True:
        scheduler(step=global_step)
        if ((global_step % config.eval_interval == 0) or (global_step >= config.max_iters)) and master_process:
            out = {}
            model.eval()
            for split in ["train", "val"]:
                batch_iter = iter_batches(split=split)
                losses = torch.zeros(config.eval_iters)  # keep on CPU
                for i in range(config.eval_iters):
                    with torch.no_grad():
                        X, Y = next(batch_iter)
                        with ctx:
                            _, loss = model(X, Y)
                            loss = loss.item()
                    losses[i] = loss
                out[f"{split}"] = losses.mean()
            print(
                f"global_step: {global_step}, train_loss: {out['train']:.4f}, val_loss: {out['val']:.4f}"
            )
            if config.wandb_log:
                try:
                    wandb.log(
                        {
                            "global_step": global_step,
                            "train_loss": out["train"],
                            "val_loss": out["val"],
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    )
                except:
                    print("Failed to log to wandb")
            model.train()
            if out["val"] < best_val_loss or config.always_save_checkpoint:
                best_val_loss = out["val"]
                if global_step > 0:
                    print(f"Saving checkpoint at global_step: {global_step}")
                    save_checkpoint(
                        config.out_dir,
                        model,
                        optimizer,
                        model_args,
                        global_step,
                        best_val_loss,
                        config,
                        lora=config.init_from == "lora",
                    )
        if global_step == 0 and config.eval_only:
            break

        for micro_step in range(config.gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == config.gradient_accumulation_steps - 1
                )
            with ctx:
                _, loss = model(X, Y)
                loss = model.last_loss
                loss = loss / config.gradient_accumulation_steps
            # get next batch while model does forward pass
            X, Y = next(train_batch_iter)
            # backward pass
            scaler.scale(loss).backward()
        # gradient clipping
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        # step the optimizer and scaler
        scaler.step(optimizer)
        scaler.update()
        # flush gradients
        optimizer.zero_grad(set_to_none=True)

        if global_step % config.log_interval == 0 and master_process:
            lossf = loss.item() * config.gradient_accumulation_steps
            lr = optimizer.param_groups[0]["lr"]
            print(f"{global_step} | loss: {lossf:.4f} | lr: {lr:.2e}")
            if config.wandb_log:
                try:
                    wandb.log(
                        {
                            "global_step": global_step,
                            "train_loss": lossf,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    )
                except:
                    print("Failed to log to wandb")

        global_step += 1

        if global_step >= config.max_iters:

            print(" *---- Finished training ----* ")
            print(f"Saving checkpoint at global_step: {global_step}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"Final iteration: {global_step}")
            break

    # save final checkpoint
    if master_process:
        save_checkpoint(
            config.out_dir,
            model,
            optimizer,
            model_args,
            global_step,
            best_val_loss,
            config,
            lora=config.init_from == "lora",
        )


def main():
    # validating checks
    assert config.vocab_source in ["llama2", "custom"]
    assert (
        config.vocab_source == "custom" or config.vocab_size == 32000
    ), "The vocab from Meta has 32K tokens"

    ddp, master_process, seed_offset, ddp_world_size, ddp_local_rank = setup_ddp()
    tokens_per_iter = (
        config.gradient_accumulation_steps
        * ddp_world_size
        * config.batch_size
        * config.max_seq_len
    )
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        print(
            f"breaks down as: {config.gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {config.batch_size} batch size * {config.max_seq_len} max seq len"
        )
        # create output directory
        print(f"Creating output directory: {config.out_dir}")
        os.makedirs(config.out_dir, exist_ok=True)
        if config.wandb_log:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config,
            )

    # reproducibility
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in config.device else "cpu"
    )  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # task-specific setup
    iter_batches = partial(
        Task.iter_batches,
        repo=config.dataset_repo,
        data_dir=config.dataset_directory,
        batch_size=config.batch_size,
        eval_iters=config.eval_iters,
        max_seq_len=config.max_seq_len,
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
    model = init_model(config.init_from, model_args, config)
    # wrap model into DDP container
    if ddp:
        # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
        # construction time since NCCL does not support `ComplexFloat`
        prefix = "_orig_mod." if compile else ""
        model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
        model = DDP(model, device_ids=[ddp_local_rank])
    model.to(config.device)

    # optimizer
    optimizer = model.configure_optimizers(
        config.weight_decay,
        config.learning_rate,
        (config.beta1, config.beta2),
        config.device_type,
    )
    scheduler = lr_scheduler(
        optimizer,
        config.learning_rate,
        config.warmup_iters,
        config.max_iters,
        config.min_lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == "float16"))

    # training loop
    train(
        model,
        optimizer,
        scheduler,
        scaler,
        iter_batches,
        model_args,
        ctx,
        master_process,
        ddp,
    )

    if ddp:
        destroy_process_group()

    print(" *---- Finished training ----* ")


if __name__ == "__main__":
    main()
