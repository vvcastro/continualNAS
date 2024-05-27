import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from typing import Callable, Dict

from .utils import metric_smoothing


class OFAModelTrainer:
    """
    Class to handle training of a model with custom metrics tracking.

    Attributes:
    model (torch.nn.Module): The model to be trained.
    criterion (Callable): Loss function.
    optimizer (torch.optim.Optimizer): Optimizer for training.
    custom_metrics (Dict[str, Callable]): Dictionary of custom metrics to track during training.
    device (str): Device to run training on, default is 'cuda'.
    """

    def __init__(self, model: nn.Module, custom_metrics: Dict[str, Callable] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Track training metrics
        self.custom_metrics = custom_metrics if custom_metrics else {}
        self.metrics_history = {
            "train_loss": [],
            "val_loss": [],
            **{f"train_{metric}": [] for metric in custom_metrics},
            **{f"val_{metric}": [] for metric in custom_metrics},
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimiser: optim.Optimizer,
        criterion: Callable,
        epochs: int = 10,
    ):
        """
        Train the model and validate at the end of each epoch, tracking custom metrics.

        Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of epochs to train for.
        """
        for epoch in tqdm(range(epochs), desc="Training"):
            self.train_epoch(train_loader, optimiser, criterion)
            self.validate_epoch(val_loader, criterion)

            self.print_epoch_metrics(epoch, phase="train")
            self.print_epoch_metrics(epoch, phase="val")

    def train_epoch(
        self,
        data_loader: DataLoader,
        optimiser: optim.Optimizer,
        criterion: Callable,
    ):
        """
        Perform a single training epoch.

        Args:
        data_loader (DataLoader): DataLoader for training data.
        """
        self.model.train()

        _losses, _metrics = [], {metric: [] for metric in self.custom_metrics}
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Optimisation step
            optimiser.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            # Track metric results
            _losses.append(loss.item())
            for metric_name, metric_fn in self.custom_metrics.items():
                _metrics[metric_name].append(metric_fn(outputs, labels).item())

        # Add metrics grouped by epoochs
        self.metrics_history["train_loss"].append(_losses)
        for metric_name, metric_values in _metrics.items():
            self.metrics_history[f"train_{metric_name}"].append(metric_values)

    def validate_epoch(self, data_loader: DataLoader, criterion: Callable):
        """
        Perform a single validation epoch.

        Args:
        data_loader (DataLoader): DataLoader for validation data.
        """
        self.model.eval()

        # Iterate over the dataset without gradients tracking
        _losses, _metrics = [], {metric: [] for metric in self.custom_metrics}
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Compute metrics
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Store batch metrics
                _losses.append(loss.item())
                for metric_name, metric_fn in self.custom_metrics.items():
                    _metrics[metric_name].append(metric_fn(outputs, labels).item())

        # Add metrics grouped by epoochs
        self.metrics_history["val_loss"].append(_losses)
        for metric_name, metric_values in _metrics.items():
            self.metrics_history[f"val_{metric_name}"].append(metric_values)

    def plot_metrics(self):
        """
        Plot the tracked metrics over epochs as subplots.
        """
        num_metrics = len(self.custom_metrics) + 1
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))

        if num_metrics == 1:
            axes = [axes]

        for idx, metric_name in enumerate(["loss"] + list(self.custom_metrics.keys())):
            ax = axes[idx]

            train_metric = self.metrics_history[f"train_{metric_name}"]
            val_metric = self.metrics_history[f"val_{metric_name}"]

            # Plot split metrics maintaining colors
            flattened_train = np.concatenate(train_metric, axis=0)
            flattened_val = np.concatenate(val_metric, axis=0)
            diff_factor = len(flattened_train) / len(flattened_val)

            # Comute the ranges for the plotting
            train_range = np.arange(len(flattened_train))
            val_range = np.arange(len(flattened_val)) * diff_factor

            ax.plot(train_range, flattened_train, label="Train")
            ax.plot(val_range, flattened_val, label="Validation")

            ax.set_title(f"{metric_name} Over Epochs")
            ax.set_ylabel(metric_name)
            ax.set_xlabel("Epochs")
            ax.legend()

        plt.tight_layout()
        plt.show()

    def print_epoch_metrics(self, epoch: int, phase: str = "train"):
        """
        Print metrics for a given epoch.

        Args:
        epoch (int): Epoch number.
        phase (str): Phase of metrics to print ('train' or 'val').
        """
        loss_key = f"{phase}_loss"
        metrics_keys = [
            key
            for key in self.metrics_history
            if key.startswith(phase) and key != loss_key
        ]

        # Aggregate metrics for the epoch
        aggregated_loss = sum(self.metrics_history[loss_key][-1]) / len(
            self.metrics_history[loss_key][-1]
        )
        aggregated_metrics = {
            k: sum(self.metrics_history[k][-1]) / len(self.metrics_history[k][-1])
            for k in metrics_keys
        }

        metrics_str = ", ".join(
            [f"{key}: {aggregated_metrics[key]:.6f}" for key in metrics_keys]
        )
        print(
            f"Epoch {epoch + 1}, {phase.capitalize()} Loss: {aggregated_loss:.6f}, Metrics: {metrics_str}"
        )
