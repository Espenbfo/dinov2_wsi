from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, Dataset
from torchvision import transforms
from monai.apps.pathology.transforms.stain import NormalizeHEStains
from PIL import Image
import numpy as np
import torch

root = Path("/home/espenbfo/datasets/BACH")
train_root = root / "ICIAR2018_BACH_Challenge/Photos"
test_root = root / "ICIAR2018_BACH_Challenge_TestDataset/Photos"

normstain = NormalizeHEStains()
transf = transforms.Compose([
    transforms.ToTensor(),
            transforms.RandomResizedCrop((224, 224), scale=(0.25, 1),ratio=(1,1),antialias=False),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

test_transf = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop((1536, 1536)),
            transforms.Resize((224, 224), antialias=False)
        ])

class SubsetWithTransforms(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def get_bach_datasets():
    train_and_val = ImageFolder(train_root)
    #test = ImageFolder(test_root, transform=transf)
    generator = torch.Generator()
    generator.manual_seed(1)
    train, val, test = random_split(train_and_val, [0.6, 0.1, 0.3], generator)
    train = SubsetWithTransforms(train, transf)
    val = SubsetWithTransforms(val, test_transf)
    test = SubsetWithTransforms(test, test_transf)


    train.classes = train_and_val.classes
    
    full_training = SubsetWithTransforms(train_and_val, transf)
    full_training.classes = train_and_val.classes
    return full_training, full_training, full_training
    return train, val, test

class TestDataset(Dataset):
    def __init__(self, path: Path, transforms) -> None:
        self.path = path
        self.files = list(path.rglob("*.tif"))
        print(self.files)
        self.transforms = transforms
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        image = Image.open(self.path / f"test{index}.tif")
        image = np.array(image)
        return self.transforms(image)
    
def get_bach_test_dataset():
    dataset = TestDataset(test_root, test_transf)
    return dataset