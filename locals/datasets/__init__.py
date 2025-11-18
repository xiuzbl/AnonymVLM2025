from utils.util import instantiate_from_config
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from functools import partial
import torch
import torch.distributed as dist
import transformers
from typing import Optional, Dict, Sequence
from constants import PRECISION, IGNORE_TOKEN_ID
from dataclasses import dataclass
import random


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, k, dataset, processor=None):
        self.ds_key = k
        self.data = dataset
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.processor is not None:
            item = self.processor(self.data[idx])
        else:
            item = self.data[idx]
        # item['item_id'] = '{}_{}'.format(self.ds_key, idx)
        return item