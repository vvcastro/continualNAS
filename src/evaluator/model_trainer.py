from typing import Callable, Dict
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch


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

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        optimizer: optim.Optimizer,
        custom_metrics: Dict[str, Callable] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer

        self.custom_metrics = custom_metrics if custom_metrics else {}

    def train(self, train_loader: DataLoader, epochs: int = 5):
        """
        Train the model and track custom metrics.

        Args:
        train_loader (DataLoader): DataLoader for training data.
        epochs (int): Number of epochs to train for.
        """

        # Set the model on training mode
        self.model.train()

        for epoch in tqdm(range(epochs), desc="Training model"):
            running_loss = 0.0
            metrics_results = {metric: 0.0 for metric in self.custom_metrics}
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # Performs the optimisation step
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Compute metrics
                running_loss += loss.item()
                for metric_name, metric_fn in self.custom_metrics.items():
                    metrics_results[metric_name] += metric_fn(outputs, labels).item()

            # Aggregate metrics for the whole epoch
            epoch_loss = running_loss / len(train_loader)
            epoch_metrics = {
                metric: result / len(train_loader)
                for metric, result in metrics_results.items()
            }
            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}, Metrics: {epoch_metrics}"
            )

    def show_training_results(self):
        pass
