"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""
import os
from functools import partial

import torch
from datasets import load_dataset

from gimli.tokenizer import Tokenizer


def download(repo: str, data_dir: str):
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        os.makedirs(data_dir, exist_ok=True)
        dataset = load_dataset(repo, split="train", cache_dir=data_dir)
    else:
        dataset = load_dataset(repo, split="train", cache_dir=data_dir)
    return dataset

def tokenize_example(example, tokenizer):
    """Tokenizes a single example, returns a list of tokens"""
    text = f"""### Instruction:\n{example["instruction"]}\n"""
    if example["input"] != "":
        text += f"""### Input:\n{example["input"]}\n"""
    text += f"""### Response:\n{example["output"]}"""
    text = text.strip()  # get rid of leading/trailing whitespace
    tokens = tokenizer.encode(text, bos=True, eos=False)  # encode the text, use BOS
    return {"tokens": tokens}

def preprocess_dataset(repo, max_seq_len, data_dir):
    tokenizer = Tokenizer()
    # pretokenize the dataset
    dataset = download(repo, data_dir)
    dataset = dataset.map(
        partial(tokenize_example, tokenizer=tokenizer),
        batched=False,
        remove_columns=["instruction", "input", "output"],
    )
    dataset = dataset.filter(lambda x: len(x["tokens"]) <= max_seq_len)
    return dataset


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, split, repo, max_seq_len, data_dir, batch_size, eval_iters, **kwargs):
        self.dataset = preprocess_dataset(repo, max_seq_len, data_dir)
        self.split = split
        self.val_size = batch_size * eval_iters
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        # reserve the first val_size examples for validation
        self.dataset = self.dataset.shuffle(seed=42)
        self.val_dataset = self.dataset.select(range(self.val_size))
        self.dataset = self.dataset.select(range(self.val_size, len(self.dataset)))

    def __len__(self):
        if self.split == "train":
            return len(self.dataset)
        elif self.split == "val":
            return len(self.val_dataset)
        else:
            raise ValueError(f"Unknown split {self.split}")
        
    def __iter__(self):
        if self.split == "train":
            return self.train_iter()
        elif self.split == "val":
            return self.val_iter()
        else:
            raise ValueError(f"Unknown split {self.split}")
        
    """ 
    m = np.memmap(shard, dtype=np.uint16, mode="r")
    num_batches = len(m) // self.max_seq_len
    num_batches -= 1  # drop the last partial batch
    assert num_batches > 0, "this shard is way too small? investigate."
    ixs = list(range(num_batches))
    rng.shuffle(ixs)
    for ix in ixs:
        start = ix * self.max_seq_len
        end = start + self.max_seq_len + 1
        # calling .astype will copy the data into a new numpy array, now in RAM
        chunk = torch.from_numpy((m[start:end]).astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        yield x, y

    Okay so this is reading a file of text, it's determining how many batches are in the file by max_seq_len
    Then it's shuffling the batches
    For each batch, it's reading the batch from the file, and then splitting it into x and y
    """
        
    def train_iter(self):
        while True:
            # we don't need to read a file, we just need to shuffle the dataset
            self.dataset = self.dataset.shuffle(seed=42)
            # instead of handcreating batches, we can use the DataLoader, so we just need to get an example
            for example in self.dataset:
                X = torch.tensor(example["tokens"][:-1])
                Y = torch.tensor(example["tokens"][1:])
                # we should pad each example to max_seq_len
                X = torch.nn.functional.pad(X, (0, self.max_seq_len - len(X)))
                Y = torch.nn.functional.pad(Y, (0, self.max_seq_len - len(Y)))
                yield X, Y

    def val_iter(self):
        while True:
            # we don't need to read a file, we just need to shuffle the dataset
            self.val_dataset = self.val_dataset.shuffle(seed=42)
            # instead of handcreating batches, we can use the DataLoader, so we just need to get an example
            for example in self.val_dataset:
                X = torch.tensor(example["tokens"][:-1])
                Y = torch.tensor(example["tokens"][1:])
                # we should pad each example to max_seq_len
                X = torch.nn.functional.pad(X, (0, self.max_seq_len - len(X)))
                Y = torch.nn.functional.pad(Y, (0, self.max_seq_len - len(Y)))
                yield X, Y

class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = Dataset(**dataset_kwargs, batch_size=batch_size)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y