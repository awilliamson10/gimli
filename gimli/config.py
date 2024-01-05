import sys
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from gimli.tokenizer import TokenizerConfig


@dataclass
class TrainConfiguration:
    """
    A dataclass that holds the configuration for training.
    """

    # Data related configs
    batch_size: int = 128
    max_seq_len: int = 2048
    vocab_source: str = "llama2"
    vocab_size: int = 32000

    # Model
    dim: int = 640
    n_layers: int = 10
    n_heads: int = 10
    n_kv_heads: int = 10
    multiple_of: int = 32
    dropout: float = 0.0

    # AdamW Optimizer
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-4
    max_iters: int = 100
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Learning rate decay settings
    decay_lr: bool = True
    warmup_iters: int = 1000

    # System
    device: str = "cpu"
    dtype: str = "bfloat16"
    compile: bool = True
    num_processes: int = 1

    # Training
    out_dir: str = "out"
    eval_interval: int = 0
    log_interval: int = 1
    eval_iters: int = 0
    eval_only: bool = False
    always_save_checkpoint: bool = False
    init_from: str = "scratch" # or "checkpoint" or "lora"
    wandb_log: bool = True
    wandb_project: str = "gimli_math"
    wandb_run_name: Optional[str] = None

    # LoRA
    lora_target_modules = ["wq", "wk", "wv", "wo"]
    lora_rank: int = 4
    lora_dropout: float = 0.0
    lora_alpha: float = 1.0

    # Dataset
    dataset_repo: str = ""
    dataset_dir: str = "data"
    chunk_size: int = 2049 * 1024 # 2048 block size + 1 for causal (from LLama), 1024 blocks

    @property
    def tokenizer_config(self):
        return TokenizerConfig()

    @property
    def dataset_directory(self):
        # combine dataset_dir with out_dir
        return Path(self.out_dir) / Path(self.dataset_dir)

    @property
    def lr_decay_iters(self):
        return self.max_iters

    @property
    def min_lr(self):
        return 0.0

    @property
    def device_type(self):
        return "cuda" if self.device.startswith("cuda") else "cpu"


def from_yaml_to_train_config(yaml_file):
    with open(yaml_file, "r") as f:
        config_dict = yaml.safe_load(f)
        assert isinstance(config_dict, dict)
        return TrainConfiguration(**config_dict)


def override_train_config_with_args(train_config, args):
    for arg in args:
        if arg.startswith("--"):
            key_val_pair = arg[2:].split("=")
            if len(key_val_pair) == 2:
                key, val = key_val_pair
                if hasattr(train_config, key):
                    old_val = getattr(train_config, key)
                    new_val = literal_eval(val) if not isinstance(old_val, str) else val
                    if isinstance(new_val, type(old_val)):
                        setattr(train_config, key, new_val)
                    else:
                        raise TypeError(
                            f"Type of {key} should be {type(old_val)}, but got {type(new_val)}"
                        )
                else:
                    raise ValueError(f"Unknown config key: {key}")
            else:
                raise ValueError(f"Malformed argument: {arg}")
        else:
            raise ValueError(f"Unknown argument format: {arg}")


# Parsing command line arguments
config = TrainConfiguration()
for arg in sys.argv[1:]:
    if arg.startswith("--config="):
        config_file = arg.split("=")[1]
        print(f"Loading configuration from {config_file}")
        config = from_yaml_to_train_config(config_file)
    else:
        override_train_config_with_args(config, [arg])

global_config = config
