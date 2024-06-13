from torch.utils.data import DataLoader, random_split, Dataset
import torch


from src.data.preparation import (
    transform_dataset,
)

from typing import Dict


class DataContinualDataset:
    def __init__(self, dataset: Dataset, params: Dict):
        self.base_dataset = dataset

        # How to split data params
        _split_sizes = params.get("split_sizes", [0.2] * 5)
        self.split_sizes = [s / sum(_split_sizes) for s in _split_sizes]

        # Split into continual datasets
        _seed = params.get("random_seed", 42)
        _permutator = torch.Generator().manual_seed(_seed)
        self.data_splits = random_split(
            self.base_dataset,
            self.split_sizes,
            generator=_permutator,
        )

    def prepare_split_loaders(self, batch_size: int, img_size: int, augment: bool):
        """
        Applies the training transformation to the data splits and returns them as
        DataLoaders.
        """
        _splits = [transform_dataset(d, img_size, augment) for d in self.data_splits]
        return [DataLoader(s, batch_size, shuffle=True) for s in _splits]


class TransformedDataset(Dataset):
    """
    Custom class to implement transformation in `Subsets` of the dataset.
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
