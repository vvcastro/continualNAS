from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import Subset

from typing import List, Tuple


type Dataset = datasets.CIFAR10 | datasets.CIFAR100


def stratified_train_test_split(dataset: Dataset, train_size: float = 0.8):
    """
    Split the dataset into train and test sets using a shuffled stratified approach.

    Args:
    dataset: The dataset to split.
    train_size (float): Proportion of the dataset to include in the train split.

    Returns:
    [Subset, Subset]: Train and test subsets.
    """
    targets = dataset.targets
    train_indices, test_indices = train_test_split(
        range(len(targets)), train_size=train_size, stratify=targets, random_state=42
    )
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    return train_subset, test_subset


def select_classes(dataset: Dataset, classes: List[int]) -> Subset:
    """
    Select only certain classes from the dataset.

    Args:
    dataset: The dataset to filter.
    classes (List[int]): List of class indices to include.

    Returns:
    Subset: Filtered subset of the dataset.
    """
    class_indices = [i for i, label in enumerate(dataset.targets) if label in classes]
    subset = Subset(dataset, class_indices)
    return subset


def transform_dataset(dataset: Dataset, resolution: Tuple[int, int]) -> Dataset:
    """
    Prepare the dataset by applying transformations.

    Args:
    dataset: The dataset to transform.
    input_resolution (int): The resolution to resize the images to.

    Returns:
    datasets.CIFAR10: Transformed dataset.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset.transform = transform
    return dataset
