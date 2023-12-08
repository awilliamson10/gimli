from transformers import PreTrainedTokenizerBase
from typing import List, Dict
import torch
import functools
from datasets import load_dataset, interleave_datasets
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import os
import numpy as np

def process_shard(
    tokenizer: PreTrainedTokenizerBase, max_tokens: int, examples: List[str]
) -> Dict[str, List]:
    res = tokenizer(
        examples,
        truncation=True,
        max_length=max_tokens - 2,
        add_special_tokens=True,
    )
    token_ids = [torch.tensor(seq) for seq in res["input_ids"]]
    return {
        "token_ids": token_ids,
    }


def load_pretraining_dataset(
    path, tokenizer, max_seq_len=2048, seed=42, interleave=False, probs=[]
):
    if interleave and len(probs) != len(path):
        raise ValueError("ratio must have the same length as path")
    encode = functools.partial(process_shard, tokenizer, max_seq_len)
    if interleave:
        if not isinstance(probs, list):
            probs = [probs]
        datasets = []
        for p in path:
            dataset = load_dataset(p, streaming=True, split="train")
            datasets.append(dataset)
        dataset = interleave_datasets(datasets, probabilities=probs, seed=seed)
    else:
        dataset = load_dataset(path, streaming=True, split="train")
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    dataset = dataset.map(
        encode,
        batched=True,
        input_columns="text",
        # remove all the existing columns after mapping since they end up having
        # a different length than the encoded/tokenized column
        remove_columns=dataset.features.keys(),
    )
    return dataset


def collate_fn(batch):
    inputs = [item["token_ids"] for item in batch]
    inputs = pad_sequence(inputs, batch_first=True)
    targets = inputs.clone()  # copy inputs to targets
    return {"input_ids": inputs[:, :-1], "labels": targets[:, 1:]}


class Task:
    @staticmethod
    def prepare_data(save_dir, steps, val_size, **kwargs):
        dataset = load_pretraining_dataset(**kwargs)
        train_dataset, val_dataset = random_split(dataset, [steps, val_size])
        with open(os.path.join(save_dir, 'train_data.bin'), 'wb') as f:
            np.array(train_dataset, dtype=np.uint16).tofile(f)
        with open(os.path.join(save_dir, 'val_data.bin'), 'wb') as f:
            np.array(val_dataset, dtype=np.uint16).tofile(f)

    @staticmethod
    def iter_batches(batch_size, device, num_workers, split='train', **dataset_kwargs):
        data_path = os.path.join(dataset_kwargs['dir'], f'{split}_data.bin')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'{data_path} not found')
        dataset = np.memmap(data_path, dtype=np.uint16, mode='r')
        dataset = torch.from_numpy(dataset.astype(np.int64))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        for batch in dataloader:
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            yield x, y

