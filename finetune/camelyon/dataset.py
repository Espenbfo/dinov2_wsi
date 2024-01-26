from torch.utils.data import Dataset
from pathlib import Path
from monai.data import WSIReader
import torch
import numpy as np

class CamyleonDataset(Dataset):
    def __init__(self, images_root: str | Path, masks_root: str | Path) -> None:
        super().__init__()
        self.images_root = Path(images_root)
        self.masks_root = Path(masks_root)
        self.files = self.find_valid_files()
        self.reader = WSIReader(backend="cucim")
        self.reader_mask = WSIReader(backend="tifffile") ## Cucim doesn't support non-rgb images

    
    def find_valid_files(self):
        masks = list(self.masks_root.rglob("**/*.tif"))
        images = list(self.images_root.rglob("**/*.tif"))
        image_stems = [file.stem for file in images]
        mask_stems = [file.stem.replace("_mask", "") for file in masks]

        # Make sure that each mask has a corresponding image
        masks = sorted(list(filter(lambda filename: (filename.stem.replace("_mask", "") in image_stems), masks)))
        # Make sure that each image has a corresponding mask
        images = sorted(list(filter(lambda filename: filename.stem in mask_stems, images)))

        return {"masks": masks, "images": images}

    def __len__(self):
        return len(self.files["images"])
    
    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        mask_filename = self.files["masks"][index]
        image_filename = self.files["images"][index]

        print(mask_filename)
        print(image_filename)
        mask_file = self.reader_mask.read(mask_filename.as_posix())
        image_file = self.reader.read(image_filename.as_posix())

        mask_highest_level = self.reader_mask.get_level_count(mask_file) - 1
        
        mask_patch = self.reader_mask.get_data(mask_file, level=mask_highest_level, mode="Å")
        
        image_highest_level = self.reader.get_level_count(image_file) - 1

        image_patch = self.reader.get_data(image_file, level=image_highest_level)


        
        return (image_patch[0], mask_patch[0])
    