from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import *
from torch.utils.data import random_split
import torch
def load_datasets(folder: Path, target_size=(224,224), train_fraction=0.9):
    dataset =  ImageFolder(folder, transform=Compose([Resize(target_size), RandomHorizontalFlip(), ToTensor()]))
    if (train_fraction == 1):
        return dataset, None, dataset.classes
    generator = torch.Generator().manual_seed(43)
    return *random_split(dataset, (train_fraction, 1-train_fraction), generator), dataset.classes

def load_dataloader(dataset, batch_size, classes, shuffle=True):
    return DataLoader(dataset, batch_size, shuffle, num_workers=6)