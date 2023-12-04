from monai.data import WSIReader
from torch.utils.data import Dataset
from pathlib import Path
import random
import numpy as np
import cv2
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor
from PIL import Image

class WSIDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        transforms = None,
        transform = None,
        target_transform=None,
        min_physical_size=100,
        max_physical_size=1000,
        base_resolution=1024,
    ):
        super().__init__(root=root)
        self.reader = WSIReader()
        path = Path(root)
        self.files = list(path.rglob("*.svs"))
        self.transform = transform
        self.samples_pr_slide_pr_epoch = 64
        self.base_resolution = base_resolution
        self.min_physical_size = min_physical_size
        self.max_physical_size = max_physical_size
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.files)*self.samples_pr_slide_pr_epoch

    def __getitem__(self, index):
        index //= self.samples_pr_slide_pr_epoch
        path = self.files[index]
        physical_size = np.random.randint(
            self.min_physical_size, self.max_physical_size
        )
        patch = self.random_valid_patch(path, physical_size, self.base_resolution)
        target = self.get_target(index)
        return self.transform(Image.fromarray(patch)), target

    def extract_valid_patches(self, wsi, patch_physical_size, threshold=0.1):
        highest_level = self.reader.get_level_count(wsi) - 1
        microns_prpx_x, microns_prpx_y = self.reader.get_mpp(wsi, highest_level)

        whole_image, meta_data = self.reader.get_data(wsi, level=highest_level)
        width_pixels, height_pixels = meta_data["spatial_shape"]
        width = width_pixels * microns_prpx_x
        height = height_pixels * microns_prpx_y
        n_patches_width = int(width // patch_physical_size)
        n_patches_height = int(height // patch_physical_size)

        assert n_patches_width > 0 and n_patches_height > 0

        whole_image = whole_image.mean(axis=0)

        resized = cv2.resize(whole_image, (n_patches_width, n_patches_height))

        resized = (resized < 255 - threshold * 255).astype(int)

        return resized

    def get_patch_at_location(
        self, wsi, location, patch_physical_size, resolution, location_is_center=False
    ):
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
        image = cv2.resize(np.moveaxis(image_before, 0, 2), (resolution, resolution))

        return image

    def random_valid_patch(self, path, patch_physical_size, resolution):
        wsi = self.reader.read(path.as_posix())

        valid_patches = self.extract_valid_patches(wsi, patch_physical_size)
        options = np.argwhere(valid_patches > 0)

        rnd_index = np.random.choice(len(options))
        choice = options[rnd_index]

        exact_pos = (
            choice[0] / valid_patches.shape[0],
            choice[1] / valid_patches.shape[1],
        )

        return self.get_patch_at_location(
            wsi, exact_pos, patch_physical_size, resolution
        )

    def random_slide(self, base_res=1024):
        path = random.choice(self.files)
        physical_size = np.random.randint(
            self.min_physical_size, self.max_physical_size
        )
        patch = self.random_valid_patch(path, physical_size, base_res)
        return patch

    @staticmethod
    def roll_channels(slide):
        return np.moveaxis(slide, 0, 2)

    def get_target(self, index: int):
        return 0
    
