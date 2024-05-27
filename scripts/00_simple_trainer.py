from _configs import CIFAR10_DATA_DIR, OFA_MODEL_PATH

from src.search_space.ofa_space import OFASearchSpace
from src.evaluator.model_trainer import OFAModelTrainer

from src.evaluator.evaluator import OFAEvaluator
from src.data.preparation import (
    select_classes,
    transform_dataset,
)

from src.evaluator.metrics import binary_accuracy


from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import (
    DynamicSeparableConv2d,
)

from torch.utils.data import DataLoader
from torchvision import datasets
import torch.optim as optim
import torch.nn as nn

DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1


ofa_net = OFAEvaluator(
    family="mobilenetv3",
    model_path=OFA_MODEL_PATH,
    pretrained=True,
    data_classes=2,
)

# Step 0: Sample an architecture
search_space = OFASearchSpace(family="mobilenetv3")
sampled_architecture = search_space.sample(n_samples=1)[0]
model, _ = ofa_net.get_architecture_model(sampled_architecture)

# Step 1. Load CIFAR-10 train/test dataset
print("(1) Preparing datasets...")
train_dataset = datasets.CIFAR10(root=CIFAR10_DATA_DIR, train=True, download=True)
test_dataset = datasets.CIFAR10(root=CIFAR10_DATA_DIR, train=False, download=True)

# Step 2. Add the same transformation to the dataset
train_dataset = transform_dataset(train_dataset, sampled_architecture["resolution"])
prepared_dataset = transform_dataset(test_dataset, sampled_architecture["resolution"])

# Step 3: Select specific classes for continual learning
train_dataset = select_classes(train_dataset, classes=[0, 1])
test_dataset = select_classes(test_dataset, classes=[0, 1])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 5: Train the model
print("(2) Training the model...")
metrics = {"accuracy": binary_accuracy}
trainer = OFAModelTrainer(model, custom_metrics=metrics)

optimiser = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
trainer.train(
    train_loader,
    val_loader,
    optimiser,
    criterion,
    epochs=1,
)

# Show training metrics.
trainer.plot_metrics()
