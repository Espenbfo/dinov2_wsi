import argparse
from pathlib import Path
import tqdm
import h5py
from monai.data import WSIReader
import cv2
import os
import numpy as np

MIN_PATCH_SIZE = 100  # microns
THRESHOLD = 0.1
LEVEL = 6


def main():
    parser = argparse.ArgumentParser(
        prog="WSI Thumbnail extractor preprocessing", description="Extracts thumbnails from WSIs"
    )

    parser.add_argument("dataset_folder_images", help="Slide files parent directory")
    parser.add_argument("dataset_folder_mask", help="Masked files parent directory")
    parser.add_argument("output_filename")
    parser.add_argument("-o", "--overwrite", action="store_true")

    args = parser.parse_args()

    images_root = Path(args.dataset_folder_images)
    masks_root = Path(args.dataset_folder_mask)
    output_filename = Path(args.output_filename)
    overwrite = args.overwrite

    wsi_reader = WSIReader(backend="tifffile")

    print(output_filename)

    if output_filename.exists():
        if overwrite:
            os.remove(output_filename)
        else:
            raise FileExistsError(
                "Output file already exists and overwrite was not specified. Add -o if you want to replace the existing file"
            )

    output_file = h5py.File(output_filename.as_posix(), "w")

    masks = list(masks_root.rglob("**/*.tif"))
    images = list(images_root.rglob("**/*.tif"))
    image_stems = [file.stem for file in images]
    mask_stems = [file.stem.replace("_mask", "") for file in masks]
    # Make sure that each mask has a corresponding image
    masks = sorted(list(filter(lambda filename: (filename.stem.replace("_mask", "") in image_stems), masks)))
    # Make sure that each image has a corresponding mask
    images = sorted(list(filter(lambda filename: filename.stem in mask_stems, images)))

    for i, file in tqdm.tqdm(enumerate(masks), total=len(masks)):
        assert file.as_posix() not in output_file
        slide_file = wsi_reader.read(masks[i].as_posix())

        whole_image, meta_data = wsi_reader.get_data(slide_file, level=6, mode="hahaha")
        height_pixels, width_pixels = meta_data["spatial_shape"]
        width = width_pixels
        height = height_pixels

        whole_image_grayscale = whole_image.mean(axis=0)
        resized = cv2.resize(whole_image_grayscale, (512, 512))

        # whole_image_grayscale = (resized < 255 - THRESHOLD * 255).astype(int)
        grp = output_file.create_group(str(i))
        dataset = grp.create_dataset("thumbnail", data=resized)
        dataset.attrs["width"] = width
        dataset.attrs["height"] = height

        grp.attrs["image_file"] = images[i].as_posix()
        grp.attrs["mask_file"] = masks[i].as_posix()
        labels = np.unique(whole_image)
        grp.attrs["labels"] = tuple(labels)
        if 2 in labels:
            inner_image = whole_image[0]
            tumor_coords = np.argwhere(inner_image == 2).astype(float)
            tumor_coords[:, 0] /= inner_image.shape[0]
            tumor_coords[:, 1] /= inner_image.shape[1]
            grp.create_dataset("cancer", data=tumor_coords)
            print(images[i].as_posix())


if __name__ == "__main__":
    main()
