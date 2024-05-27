from _configs import CIFAR10_DATA_DIR, OFA_MODEL_PATH

from src.search_space.ofa_space import OFASearchSpace
from src.evaluator.model_trainer import OFAModelTrainer
from src.evaluator.model_tester import OFAModelTester
from src.evaluator.evaluator import OFAEvaluator
from src.data.preparation import (
    select_classes,
    transform_dataset,
)

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

# Step 1: Sample an architecture
search_space = OFASearchSpace(family="mobilenetv3")
sampled_architecture = search_space.sample(n_samples=1)[0]

# Initialize the model
model, _ = ofa_net.get_architecture_model(sampled_architecture)
print(model)

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
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 5: Train the model
print("(2) Training the model...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
trainer = OFAModelTrainer(model, criterion, optimizer)
trainer.train(train_loader, epochs=10)

# Step 6: Test the model
print("(3) Testing the model...")
tester = OFAModelTester(model)
accuracy, metrics = tester.test(test_loader)
print(f"Test Accuracy on classes [0, 1]: {accuracy}%, Metrics: {metrics}")
