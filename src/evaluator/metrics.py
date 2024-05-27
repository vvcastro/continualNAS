import torch


def binary_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute binary classification accuracy.

    Args:
    outputs (torch.Tensor): Model predictions.
    labels (torch.Tensor): Ground truth labels.

    Returns:
    torch.Tensor: Accuracy value.
    """
    preds = torch.argmax(outputs)
    correct = (preds == labels).float()
    accuracy = correct.sum() / len(correct)
    return accuracy
