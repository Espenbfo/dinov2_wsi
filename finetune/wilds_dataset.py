from pathlib import Path
import wilds
from torchvision import transforms
from torch.utils.data import Dataset
WILDS_PATH = "/home/espenbfo/datasets/wilds"

transf = transforms.Compose([
    transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=False),
            
        ])


class WildsDataset(Dataset):
    def __init__(self, wilds_subset):
        self.subset = wilds_subset
        self.classes = (0, 1)

    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, index):
        return self.subset[index][:2]
def get_wilds_datasets():
    full_dataset = wilds.get_dataset(root_dir=WILDS_PATH, dataset="camelyon17")
    
    return WildsDataset(full_dataset.get_subset("train", transform=transf)), WildsDataset(full_dataset.get_subset("id_val",  transform=transf)), WildsDataset(full_dataset.get_subset("test",  transform=transf))
