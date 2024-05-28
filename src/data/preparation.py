from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import Subset

from typing import List, Tuple


DatasetType = datasets.CIFAR10 | datasets.CIFAR100


def splits_continual_data(
    dataset: DatasetType, split_sizes: List[float], classes: List[int] = None
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

    # For stratified splits.
    targets = [dataset[i][1] for i in range(len(dataset))]

    subsets = []
    remaining_indices = list(range(len(targets)))
    for split_size in split_sizes[:-1]:
        split_length = int(len(targets) * split_size)
        split_indices, remaining_indices = train_test_split(
            remaining_indices,
            train_size=split_length,
            stratify=[targets[i] for i in remaining_indices],
            random_state=42,
        )
        subsets.append(Subset(dataset, split_indices))

    subsets.append(Subset(dataset, remaining_indices))
    return subsets


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

    transform = []
    if train:
        transform = [
            transforms.RandomResizedCrop(resolution, scale=(0.08, 1)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]

    transform.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ]
    )
    dataset.transform = transform
    return dataset
