import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# TRAINING PIPELINE

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding = 4),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )
])


# TESTING PIPELINE

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# DOWNLOADING DATASET

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)


# CREATING DATALOADER

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers = 0,
    pin_memory=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers = 0,
    pin_memory=False
)
