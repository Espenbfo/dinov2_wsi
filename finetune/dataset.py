from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import *
from torch.utils.data import random_split
import torch
from torchvision import transforms

class PathologyDataset(Dataset):
    def __init__(self, train_x, train_y):
        super().__init__()
        self.train_x = train_x
        self.train_y = train_y
        self.classes = ["not_cancer", "cancer"]
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224), antialias=False),
            
        ])
    def __len__(self):
        return len(self.train_x)
    
    
    def __getitem__(self, index):

        x = self.transforms(torch.moveaxis(torch.tensor(self.train_x[index]), 2, 0)/255)
        y = torch.tensor(self.train_y[index]).squeeze().long()
        # y = torch.nn.functional.one_hot(y, num_classes=2).float()
        #print(y.shape)
        return x, y
            
    


def load_datasets(folder: Path, target_size=(224,224), train_fraction=0.9):
    dataset =  ImageFolder(folder, transform=Compose([Resize(target_size), RandomHorizontalFlip(), ToTensor()]))
    if (train_fraction == 1):
        return dataset, None, dataset.classes
    generator = torch.Generator().manual_seed(43)
    return *random_split(dataset, (train_fraction, 1-train_fraction), generator), dataset.classes

def load_dataloader(dataset, batch_size, classes, shuffle=True):
    return DataLoader(dataset, batch_size, shuffle, num_workers=6)