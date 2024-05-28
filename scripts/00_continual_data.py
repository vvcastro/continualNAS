from _configs import CIFAR10_DATA_DIR, OFA_MODEL_PATH

from src.search_space.ofa_space import OFASearchSpace

from src.training.model_trainer import OFAModelTrainer
from src.training.metrics import binary_accuracy
from src.training.evaluator import OFAEvaluator
from src.training.utils import SAM

from src.data.preparation import (
    splits_continual_data,
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
test_dataset = datasets.CIFAR10(root=CIFAR10_DATA_DIR, train=False, download=True)

# Step 2. Add the same transformation to the dataset
testing_dataset = transform_dataset(test_dataset, sampled_architecture["resolution"])
training_dataset = transform_dataset(
    train_dataset,
    sampled_architecture["resolution"],
    train=True,
)

# Step 3: Select specific classes for continual learning
SPLIT_SIZES = [0.5, 0.3, 0.2]
training_splits = splits_continual_data(
    training_dataset,
    split_sizes=SPLIT_SIZES,
)
testing_splits = splits_continual_data(
    testing_dataset,
    split_sizes=SPLIT_SIZES,
)


# Step 5: Train the model
print("(2) Training the model...")
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

iterator = zip(EPOCHS, training_splits, testing_splits)
for epochs, train_split, test_split in iterator:
    training_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(test_split, batch_size=BATCH_SIZE, shuffle=False)

    trainer.train(
        training_loader,
        validation_loader,
        base_optimiser,
        criterion,
        epochs=epochs,
    )
    print("=" * 30)

# Show training metrics.
trainer.plot_metrics()
