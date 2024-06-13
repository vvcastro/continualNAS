from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch

from typing import Callable, Dict, List
from abc import ABC, abstractmethod

from src.data.preparation import (
    continual_random_splits,
    transform_dataset,
)


class ModelTrainer(ABC):
    def __init__(self, model: nn.Module):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.custom_metrics = {}

    def train(
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
        for inputs, labels in tqdm(data_loader, desc="Training"):
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

        return _losses, _metrics

    def validate(
        self,
        data_loader: DataLoader,
        criterion: Callable,
    ):
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

        return _losses, _metrics

    @abstractmethod
    def prepare_datasets(self):
        pass

    @abstractmethod
    def evaluate(self, optimiser, criterion, epochs: int):
        pass


class DataContinualTrainer(ModelTrainer):
    """
    Class to handle training of a model with a Data Continual primitive.

    Attributes:
    model (torch.nn.Module): The model to be trained.
    dataset (torch.utils.data.Dataset): The dataset to be used in the continual setting.
    custom_metrics (Dict[str, Callable]): Dict of Custom metrics to track during training.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        custom_metrics: Dict[str, Callable] = None,
    ):
        super().__init__(model)

        # Prepare the dataset
        self.base_dataset = dataset
        self.prepare_dataset()

        # Track training metrics
        self.custom_metrics = custom_metrics if custom_metrics else {}
        self.metrics_history = {
            "train_loss": [],
            **{f"train_{metric}": [] for metric in custom_metrics},
            "val_loss": defaultdict(list),
            **{f"val_{metric}": defaultdict(list) for metric in custom_metrics},
        }
        self.data_changes = []

    def prepare_datasets(self, img_size: int):
        SPLIT_SIZES = [0.5, 0.3, 0.2]
        continual_splits = {
            f"split-{i}": split
            for i, split in enumerate(
                continual_random_splits(self.base_dataset, split_sizes=SPLIT_SIZES)
            )
        }

        # Step 3. Prepare the training dataset with augmentation
        training_datasets = {
            key: transform_dataset(data, img_size, train=True)
            for key, data in continual_splits.items()
        }

        # Step 4. Build the validation data loaders
        validation_loaders = {
            key: DataLoader(
                transform_dataset(data, img_size),
                batch_size=64,
                shuffle=False,
            )
            for key, data in continual_splits.items()
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loaders: Dict[str, DataLoader],
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

        self.data_changes.append(len(self.metrics_history["train_loss"]))

        for epoch in range(epochs):
            train_losses, train_metrics = self.train_epoch(
                train_loader, optimiser, criterion
            )

            # Add metrics grouped by epoochs
            self.metrics_history["train_loss"].append(train_losses)
            for metric_name, metric_values in train_metrics.items():
                self.metrics_history[f"train_{metric_name}"].append(metric_values)

            self.print_epoch_metrics(epoch, split=None)

            for split_key, split_loader in val_loaders.items():
                losses, metrics = self.validate_epoch(split_loader, criterion)

                self.metrics_history["val_loss"][split_key].append(losses)
                for mname, mvalues in metrics.items():
                    self.metrics_history[f"val_{mname}"][split_key].append(mvalues)

                self.print_epoch_metrics(epoch, split=split_key)

    def plot_metrics(self, figsize=(8, 5), dpi=500):
        """
        Plot the tracked metrics over epochs as subplots.
        """
        num_metrics = len(self.custom_metrics) + 1
        fig, axes = plt.subplots(
            num_metrics,
            1,
            figsize=(figsize[0], figsize[1] * num_metrics),
            dpi=dpi,
        )

        if num_metrics == 1:
            axes = [axes]

        for idx, metric_name in enumerate(["loss"] + list(self.custom_metrics.keys())):
            ax = axes[idx]

            # Show training metrics
            train_metric = self.metrics_history[f"train_{metric_name}"]
            flattened_train = [np.mean(e_metric) for e_metric in train_metric]
            train_range = np.arange(len(flattened_train)) + 1
            ax.plot(train_range, flattened_train, label="Train")

            # Show validation metrics
            val_metric = self.metrics_history[f"val_{metric_name}"]
            for split_key in val_metric:
                flattened_val = [np.mean(emetric) for emetric in val_metric[split_key]]
                val_range = np.arange(len(flattened_val)) + 1
                ax.plot(val_range, flattened_val, label=split_key)

            ax.set_title(f"{metric_name} | Epochs")
            ax.set_ylabel(metric_name)
            ax.set_xlabel("Step")

            if metric_name != "loss":
                ax.set_ylim((0.65, 1))
            ax.legend()

        for change_step in self.data_changes[1:]:
            for ax in axes:
                ax.vlines(
                    x=change_step + 0.5,
                    ymin=0,
                    ymax=1,
                    color="black",
                    linestyles="dashed",
                )

        plt.tight_layout()
        plt.show()

    def print_epoch_metrics(
        self,
        epoch: int,
        split: str | None = None,
    ):
        """
        Print metrics for a given epoch.

        Args:
        epoch (int): Epoch number.
        """
        phase = "val" if split is not None else "train"
        loss_key = f"{phase}_loss"
        metrics_keys = [
            key
            for key in self.metrics_history
            if key.startswith(phase) and key != loss_key
        ]

        # Aggregate metrics for the epoch
        if phase == "val":
            losses = self.metrics_history[loss_key][split][-1]
            metrics = {k: self.metrics_history[k][split][-1] for k in metrics_keys}
        else:
            losses = self.metrics_history[loss_key][-1]
            metrics = {k: self.metrics_history[k][-1] for k in metrics_keys}

        aggregated_metrics = {k: sum(metrics[k]) / len(metrics[k]) for k in metrics}
        aggregated_loss = sum(losses) / len(losses)

        metrics_str = ", ".join(
            [f"{key}: {aggregated_metrics[key]:.6f}" for key in metrics_keys]
        )

        id_string = f"{phase.capitalize()}" + (
            f" - {split}" if split is not None else ""
        )
        print(
            f"{id_string} Epoch {epoch + 1}| Loss: {aggregated_loss:.6f}, {metrics_str}"
        )