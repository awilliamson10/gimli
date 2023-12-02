import glob
import os
import torch
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import random
import numpy as np

from torch.utils.data import IterableDataset

# current directory + /data
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")

class PretokDataset(IterableDataset):
    """
    An IterableDataset that yields pretokenized examples as PyTorch tensors.
    """
    
    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        """
        Initializes the PretokDataset instance.
        
        Args:
            split (str): Data split to use ('train' or 'test').
            max_seq_len (int): Maximum sequence length of the examples.
            vocab_size (int): Size of the vocabulary.
            vocab_source (str): Source of the vocabulary ('llama2' or 'custom').
        """
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_source = vocab_source

        # Determine the directory containing the required .bin files
        bin_dir_suffix = "TinyStories_all_data" if vocab_source == "llama2" else f"tok{vocab_size}"
        self.bin_dir = os.path.join(DATA_CACHE_DIR, bin_dir_suffix)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0

        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")

        # Find the .bin files and select the appropriate ones for the split
        shard_filenames = sorted(glob.glob(os.path.join(self.bin_dir, "*.bin")))
        if not shard_filenames:
            raise FileNotFoundError(f"No .bin files found in {self.bin_dir}")
        
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]

        # Yield batches infinitely
        return self._batch_iter(rng, shard_filenames)

    def _batch_iter(self, rng, shard_filenames):
        """
        An infinite iterator that shuffles and yields batches from the dataset.
        
        Args:
            rng (random.Random): The random number generator to shuffle shard filenames.
            shard_filenames (list): List of shard filenames to iterate over.
        """
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # Drop the last partial batch if exists
                
                if num_batches <= 0:
                    raise ValueError("Shard is too small; investigate the issue.")
                
                batch_indices = list(range(num_batches))
                rng.shuffle(batch_indices)
                for index in batch_indices:
                    start = index * self.max_seq_len
                    end = start + self.max_seq_len + 1

                    # Prepare the batch as torch tensors
                    batch_data = torch.from_numpy(m[start:end].astype(np.int64))
                    x, y = batch_data[:-1], batch_data[1:]
                    yield x, y


class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers
        )
        for x, y in dl:
            yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)