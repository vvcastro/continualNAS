from _configs import CIFAR10_DATA_DIR, OFA_MODEL_PATH
from src.search_space.ofa_space import OFASearchSpace

from src.training.trainers import OFAModelTrainer
from src.training.metrics import binary_accuracy
from src.training.evaluator import OFAEvaluator
from src.training.utils import SAM

from src.data.preparation import (
    continual_random_splits,
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


search_space = OFASearchSpace(family="mobilenetv3")
ofa_net = OFAEvaluator(
    family="mobilenetv3",
    model_path=OFA_MODEL_PATH,
    pretrained=True,
    data_classes=10,
)

# Step 0: Sample an architecture
sampled_architecture = search_space._get_min_sample()
# sampled_architecture = search_space._get_max_sample()
# sampled_architecture = search_space.sample(n_samples=1)[0]
model, _ = ofa_net.get_architecture_model(sampled_architecture)

# Step 1. Load CIFAR-10 train/test dataset
print("(1) Preparing datasets...")
train_dataset = datasets.CIFAR10(root=CIFAR10_DATA_DIR, train=True, download=True)
# valid_dataset = datasets.CIFAR10(root=CIFAR10_DATA_DIR, train=False, download=True)

# Step 2: Split the dataset for the continual learning problem
SPLIT_SIZES = [0.5, 0.3, 0.2]
continual_splits = {
    f"split-{i}": split
    for i, split in enumerate(
        continual_random_splits(train_dataset, split_sizes=SPLIT_SIZES)
    )
}

# Step 3. Prepare the training dataset with augmentation
training_datasets = {
    key: transform_dataset(data, sampled_architecture["resolution"], train=True)
    for key, data in continual_splits.items()
}


# Step 4. Build the validation data loaders
validation_loaders = {
    key: DataLoader(
        transform_dataset(data, sampled_architecture["resolution"]),
        batch_size=64,
        shuffle=False,
    )
    for key, data in continual_splits.items()
}

print("(2) Defining the model...")
metrics = {"accuracy": binary_accuracy}
trainer = OFAModelTrainer(model, custom_metrics=metrics)

##### PARAMS TO VARY
## batch_size = 64
## learning_rate = 1e-3
EPOCHS = [5, 3, 2]
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

base_optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Step 5: Train the model
for epochs, split_key in zip(EPOCHS, continual_splits.keys()):
    # Define the training data loader
    train_split = training_datasets[split_key]
    training_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)

    trainer.train(
        training_loader,
        validation_loaders,
        base_optimiser,
        criterion,
        epochs=epochs,
    )
    print("=" * 30)

# Show training metrics.
trainer.plot_metrics()
