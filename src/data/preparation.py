from torch.utils.data import Dataset
from torchvision import transforms
from copy import deepcopy
from typing import Tuple

from .datasets import TransformedDataset


def transform_dataset(
    dataset: Dataset,
    resolution: Tuple[int, int],
    augment: bool = False,
) -> Dataset:
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

    if augment:
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
    return TransformedDataset(deepcopy(dataset), transform=transformation)
