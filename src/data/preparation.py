from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, random_split
from copy import deepcopy


import torch

from typing import List, Tuple


DatasetType = datasets.CIFAR10 | datasets.CIFAR100


def continual_random_splits(
    dataset: DatasetType,
    split_sizes: List[float],
    random_seed: int = 42,
) -> List[Subset]:
    """
    Split the dataset into multiple disjoint subsets based on the given percentual sizes.

    Args:
    dataset (CIFAR10): The dataset to split.
    split_sizes (List[float]): The percentual sizes, from the original dataset, to give to each split.

    Returns:
    List[Subset]: List of filtered and split subsets.
    """

    # Ensure the split sizes sum to 1.0
    split_sizes = [size / sum(split_sizes) for size in split_sizes]

    # For replicability
    permutator = torch.Generator().manual_seed(random_seed)

    # Split the datasets
    return random_split(dataset, split_sizes, generator=permutator)


def transform_dataset(
    dataset: DatasetType,
    resolution: Tuple[int, int],
    train: bool = False,
) -> DatasetType:
    """
    Prepare the dataset by applying transformations.

    Args:
    dataset: The dataset to transform.
    input_resolution (int): The resolution to resize the images to.

    Returns:
    datasets.CIFAR10: Transformed dataset.
    """

    # Default values for CIFAR10
    norm_mean = [0.49139968, 0.48215827, 0.44653124]
    norm_std = [0.24703233, 0.24348505, 0.26158768]

    if train:
        image_transformation = [
            transforms.RandomResizedCrop(resolution, scale=(0.08, 1)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    else:
        image_transformation = [
            transforms.Resize(resolution),
        ]

    image_transformation.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ]
    )
    transformation = transforms.Compose(image_transformation)
    return TransformDataset(deepcopy(dataset), transform=transformation)


class TransformDataset(Dataset):
    """Custom class to implement transformation in `Subsets` of the dataset."""

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
