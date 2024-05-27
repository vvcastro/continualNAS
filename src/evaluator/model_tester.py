from typing import Callable, Dict

import torch
from torch.utils.data import DataLoader


class OFAModelTester:
    """
    Class to handle evaluation of a trained model with advanced metrics.

    Attributes:
    model (torch.nn.Module): The model to be evaluated.
    device (str): Device to run evaluation on, default is 'cuda'.
    custom_metrics (Dict[str, Callable]): Dictionary of custom metrics to compute during evaluation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        custom_metrics: Dict[str, Callable] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.custom_metrics = custom_metrics if custom_metrics else {}

    def train(self, test_loader: DataLoader):
        """
        Test the model and compute custom metrics.

        Args:
        test_loader (DataLoader): DataLoader for test data.
        """
        self.model.eval()
        correct = 0
        total = 0
        metrics_results = {metric: 0.0 for metric in self.custom_metrics}
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for metric_name, metric_fn in self.custom_metrics.items():
                    metrics_results[metric_name] += metric_fn(outputs, labels).item()
        accuracy = 100 * correct / total
        avg_metrics = {
            metric: result / len(test_loader)
            for metric, result in metrics_results.items()
        }
        print(f"Accuracy: {accuracy}%, Metrics: {avg_metrics}")
        return accuracy, avg_metrics
