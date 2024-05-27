from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import Subset

from typing import List, Tuple


DatasetType = datasets.CIFAR10 | datasets.CIFAR100


def prepare_continual_splits(
    dataset: DatasetType, split_sizes: List[float], classes: List[int] = None
) -> List[Subset]:
    """
    Split the dataset into multiple subsets based on the given sizes and optionally filter by classes.

    Args:
    dataset (CIFAR10): The dataset to split.
    split_sizes (List[float]): The percentual sizes, from the original dataset, to give to each split.
    classes (List[int], optional): List of class indices to include. Defaults to None.

    Returns:
    List[Subset]: List of filtered and split subsets.
    """
    if classes is not None:
        # Filter the dataset by the specified classes
        class_indices = [
            i for i, label in enumerate(dataset.targets) if label in classes
        ]
        filtered_dataset = Subset(dataset, class_indices)
    else:
        filtered_dataset = dataset

    targets = [filtered_dataset[i][1] for i in range(len(filtered_dataset))]

    # Ensure the split sizes sum to 1.0
    split_sizes = [size / sum(split_sizes) for size in split_sizes]

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
        subsets.append(Subset(filtered_dataset, split_indices))

    subsets.append(Subset(filtered_dataset, remaining_indices))
    return subsets


def transform_dataset(dataset: DatasetType, resolution: Tuple[int, int]) -> DatasetType:
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
