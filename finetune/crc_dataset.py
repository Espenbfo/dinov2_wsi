from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision import transforms

import torch

root = Path("/home/espenbfo/datasets/crc")
train_root = root / "NCT-CRC-HE-100K"
train_root_no_norm = root / "NCT-CRC-HE-100K-NONORM"
test_root = root / "CRC-VAL-HE-7K"

transf = transforms.Compose([
    transforms.ToTensor(),
            
        ])

def get_crc_datasets():
    train_and_val = ImageFolder(train_root, transform=transf)
    test = ImageFolder(test_root, transform=transf)
    generator = torch.Generator()
    generator.manual_seed(1)
    train, val = random_split(train_and_val, [0.9, 0.1], generator)
    train.classes = train_and_val.classes
    
    return train, val, test

def get_crc_datasets_no_norm():
    train_and_val = ImageFolder(train_root_no_norm, transform=transf)
    test = ImageFolder(test_root, transform=transf)
    generator = torch.Generator()
    generator.manual_seed(1)
    train, val = random_split(train_and_val, [0.9, 0.1], generator)
    train.classes = train_and_val.classes

    print(train_and_val.classes, test.classes)

    return train, val, test