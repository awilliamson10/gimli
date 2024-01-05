import glob
import os
from functools import partial
from multiprocessing import Pool
from typing import Optional, Tuple

import numpy as np
import torch
import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader

from gimli.packed_data import (CombinedDataset, PackedDataset,
                               PackedDatasetBuilder)


def create_dataloader(
    batch_size: int,
    max_seq_len: int,
    split: str,
    data_dir: str,
    datasets: Tuple[str, float],
    shuffle: bool = True,
    seed: int = 12345,
    num_workers: int = 0,
) -> DataLoader:
    data = []
    for ds, _ in datasets:
        filenames = glob.glob(os.path.join(data_dir, f"{ds}_{split}_*.bin"))
        print(f"Found {len(filenames)} files for {ds} {split} split.")
        dataset = PackedDataset(
            filenames, n_chunks=4, max_seq_len=max_seq_len, shuffle=shuffle, seed=seed,
            num_processes=1, process_rank=0,
        )
        data.append(dataset)

    if not data:
        raise RuntimeError(
            f"No data found at {data_dir}."
        )

    weights = [weight for _, weight in datasets]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    # combined_dataset = CombinedDataset(datasets=data, seed=seed, weights=weights)

    return DataLoader(data[0], batch_size=batch_size, shuffle=False, pin_memory=True)


class Task:
    @staticmethod
    def packed_prepare(tokenizer, datasets, destination_path, chunk_size, eval_iters = 0):
        destination_path.mkdir(parents=True, exist_ok=True)
        
        def create_builder(ds, split):
            parent_path = os.path.join(destination_path, ds)
            os.makedirs(parent_path, exist_ok=True)
            return PackedDatasetBuilder(
                outdir=destination_path,
                prefix=ds + "_" + split,
                chunk_size=chunk_size,
                sep_token=tokenizer.bos_token_id,
                vocab_size=tokenizer.vocab_size,
            )

        def process_dataset(ds, split, eval_iters, tokenizer, builder):
            dataset = load_dataset(ds, split=split, streaming=True)
            dataset = dataset.shuffle(seed=42)
            for i, line in tqdm.tqdm(enumerate(dataset)):
                if i < eval_iters:
                    text = line["text"]
                    text_ids = tokenizer.encode(text)
                    builder.add_array(np.array(text_ids, dtype=builder.dtype))
                if i == eval_iters:
                    builder.write_reminder()
                    builder = create_builder(ds, "train")
                if i >= eval_iters:
                    text = line["text"]
                    text_ids = tokenizer.encode(text)
                    builder.add_array(np.array(text_ids, dtype=builder.dtype))
            builder.write_reminder()

        for name, ratio in datasets:
            builder = create_builder(name, "train")
            process_dataset(name, "train", eval_iters, tokenizer, builder)


    @staticmethod
    def create_dataloaders(
        batch_size: int,
        max_seq_len: int,
        data_dir: str,
        datasets: Tuple[str, float],
        validation: bool = False,
        seed: int = 12345,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader]:
        # Increase by one because we need the next word as well
        effective_seq_len = max_seq_len + 1
        train_dataloader = create_dataloader(
            batch_size=batch_size,
            max_seq_len=effective_seq_len,
            split="train",
            data_dir=data_dir,
            datasets=datasets,
            shuffle=True,
            seed=seed,
            num_workers=num_workers,
        )
        val_dataloader = (
            create_dataloader(
                batch_size=batch_size,
                max_seq_len=effective_seq_len,
                split="val",
                data_dir=data_dir,
                datasets=datasets,
                shuffle=False,
                seed=seed,
                num_workers=num_workers,
            )
            if validation
            else None
        )
        return train_dataloader, val_dataloader


    @staticmethod
    def iter_batches(
        max_seq_len: int,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        device: torch.device,
        split: str = "train",
    ):
        dataloader = train_dataloader if split == "train" else val_dataloader
        for iter_num, data in enumerate(dataloader):
            input_ids = data[:, 0 : max_seq_len].contiguous()
            targets = data[:, 1 : max_seq_len + 1].contiguous()
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            yield input_ids, targets