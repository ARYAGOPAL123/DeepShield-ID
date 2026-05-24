import os
import random
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import torch.nn as nn


def get_transforms(img_size=224, training=True):
    if training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_dataloaders(data_dir, batch_size=16, img_size=224, val_split=0.2, seed=42):
    dataset = ImageFolder(data_dir, transform=get_transforms(img_size, training=True))
    total = len(dataset)
    if total == 0:
        raise ValueError(f"No images found in {data_dir}. Please generate or download the dataset.")
    val_len = max(1, int(total * val_split))
    train_len = total - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed))
    # Replace transforms for validation set
    val_set.dataset.transform = get_transforms(img_size, training=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, dataset.classes


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
