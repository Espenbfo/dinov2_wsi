from torch.utils.data import Dataset
from pathlib import Path
from monai.data import WSIReader
import torch
import numpy as np
import cv2
import h5py
from monai import transforms

def get_transforms(sizes, is_train):
    keys = [i for i in range(len(sizes))]
    base_transforms = [
            transforms.ToTensord(keys=keys),
            transforms.EnsureChannelFirstd(keys=keys, channel_dim=2),
            transforms.ScaleIntensityRanged(
                keys=keys, a_min=0, a_max=255, b_min=0, b_max=1, clip=True
            ),
            transforms.Resized(keys=keys, spatial_size=(224, 224), anti_aliasing=True)
        ]
    if (is_train):
        return transforms.Compose(base_transforms + [
            transforms.RandFlipd(keys=keys, prob=0.5),
            transforms.RandRotate90d(keys=keys)
        ])
    return transforms.Compose(base_transforms)

class CamyleonDataset(Dataset):
    def __init__(
        self, preprocessed_data_file: Path | str, is_train=True, sizes=(100, 400), train_fraction=0.8, iterations_per_epoch_multiplier=100
    ) -> None:
        super().__init__()
        self.preprocessed_data = h5py.File(preprocessed_data_file, "r")
        self.reader = WSIReader(backend="cucim")
        self.reader_mask = WSIReader(backend="tifffile")  ## Cucim doesn't support non-rgb images
        self.labels = (1, 2)

        self.is_train = is_train
        self.train_fraction = train_fraction

        self.sizes = sizes

        self.files = self.find_valid_files()

        self.rng_seed = 33
        self.random = np.random.default_rng(self.rng_seed) if not self.is_train else np.random

        self.transforms = get_transforms(sizes, is_train)

        self.iterations_per_epoch_multiplier = iterations_per_epoch_multiplier

        self.reset_rng()
        print(f"INIT DATASET train {self.is_train}")

    def reset_rng(self):
        self.random = np.random.default_rng(self.rng_seed) if not self.is_train else np.random

    def find_valid_files(self):
        masks = []
        images = []
        label_to_index = {label: list() for label in self.labels}

        keys = list(self.preprocessed_data.keys())
        generator = torch.Generator().manual_seed(101)
        train, val = torch.utils.data.random_split(
            range(len(keys)), [self.train_fraction, 1 - self.train_fraction], generator
        )
        data = train if self.is_train else val
        for i, index in enumerate(data):
            key = keys[index]
            images.append(self.preprocessed_data[key].attrs["image_file"])
            masks.append(self.preprocessed_data[key].attrs["mask_file"])
            for label in self.preprocessed_data[key].attrs["labels"]:
                if label in self.labels:
                    label_to_index[label].append((i, key))

        self.label_to_index = label_to_index
        return {"masks": masks, "images": images}

    def __len__(self):
        return len(self.files["images"]) * self.iterations_per_epoch_multiplier

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        label = self.random.choice(self.labels, p=(0.5, 0.5) if self.is_train else None)
        index, key = self.label_to_index[label][self.random.choice(len(self.label_to_index[label]))]
        return self.retrieve_patch_with_label(label, index, key, self.sizes), label

    def get_image_and_mask(self, index):
        mask_filename = self.files["masks"][index]
        image_filename = self.files["images"][index]
        mask_file = self.reader_mask.read(mask_filename)
        image_file = self.reader.read(image_filename)

        mask_highest_level = self.reader_mask.get_level_count(mask_file) - 2

        mask_patch = self.reader_mask.get_data(mask_file, level=mask_highest_level, mode="Å")

        image_highest_level = self.reader.get_level_count(image_file) - 2

        image_patch = self.reader.get_data(image_file, level=image_highest_level)

        return (image_patch[0], mask_patch[0])

    def find_random_area_with_label(self, mask: np.ndarray, label: int):
        valid_coords = np.argwhere(mask == label)+0.5
        random_coord = valid_coords[self.random.choice(len(valid_coords))]
        mask_shape = mask.shape

        y, x = random_coord[0] / mask_shape[0], random_coord[1] / mask_shape[1]

        return y, x

    def retrieve_patch_with_label(self, label, index, key, sizes: tuple[int]):
        mask_filename = self.files["masks"][index]
        mask_file = self.reader_mask.read(mask_filename)

        mask_highest_level = self.reader_mask.get_level_count(mask_file) - 1
        mask = self.reader_mask.get_data(mask_file, level=mask_highest_level, mode="Å")

        if label == 2:
            cancer_coords = self.preprocessed_data[key]["cancer"]
            coords = cancer_coords[self.random.choice(len(cancer_coords))]
        else:
            coords = self.find_random_area_with_label(mask[0][0], label)
        image_filename = self.files["images"][index]
        image_file = self.reader.read(image_filename)

        images = [self.get_patch_at_location(image_file, coords, size, 224, True) for size in sizes]
        image_dict = {i: image for i, image in enumerate(images)}
        image_dict = self.transforms(image_dict)
        images = [image for image in image_dict.values()]
        return images

    def get_patch_at_location(self, wsi, location, patch_physical_size, resolution, location_is_center=False):
        resolution_at_level_0 = wsi.shape
        mpp_at_zero = max(self.reader.get_mpp(wsi, level=0))
        location_coord = (
            int(resolution_at_level_0[0] * location[0]),
            int(resolution_at_level_0[1] * location[1]),
        )

        if location_is_center:
            location_coord = (
                location_coord[0] - int(patch_physical_size / mpp_at_zero / 2),
                location_coord[1] - int(patch_physical_size / mpp_at_zero / 2),
            )

        target_mpp = patch_physical_size / resolution
        current_level = 0
        mpp_at_level = mpp_at_zero
        for level in range(self.reader.get_level_count(wsi)):
            mpp = max(self.reader.get_mpp(wsi, level=level))
            if mpp > target_mpp:
                break
            current_level = level
            mpp_at_level = mpp
        patch_resolution = int(patch_physical_size / mpp_at_level)

        image_before, meta_data = self.reader.get_data(
            wsi,
            location_coord,
            size=(patch_resolution, patch_resolution),
            level=current_level,
        )
        image = cv2.resize(np.moveaxis(image_before, 0, 2), (resolution, resolution), interpolation=cv2.INTER_AREA)

        return image
