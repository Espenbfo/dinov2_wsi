import argparse
from pathlib import Path
import tqdm
import h5py
from monai.data import WSIReader
import cv2
import os

MIN_PATCH_SIZE = 100 # microns
THRESHOLD = 0.1
def main():
    parser = argparse.ArgumentParser(
                    prog='WSI Thumbnail extractor preprocessing',
                    description='Extracts thumbnails from WSIs')

    parser.add_argument("dataset_folder")
    parser.add_argument("output_filename")
    parser.add_argument("-o", "--overwrite", action="store_true")

    args=parser.parse_args()

    dataset_folder = Path(args.dataset_folder)
    output_filename= Path(args.output_filename)
    overwrite = args.overwrite

    wsi_reader = WSIReader()
    print(dataset_folder)
    print(output_filename)

    if (output_filename.exists()):
        if (overwrite):
            os.remove(output_filename)
        else:
            raise FileExistsError("Output file already exists and overwrite was not specified. Add -o if you want to replace the existing file")
    
    output_file = h5py.File(output_filename.as_posix(), "w")

    files = list(dataset_folder.rglob("*.svs"))

    for file in tqdm.tqdm(files):
        assert file.as_posix() not in output_file
        
        slide_file = wsi_reader.read(file.as_posix())
        highest_level = wsi_reader.get_level_count(slide_file) - 1
        microns_prpx_y, microns_prpx_x = wsi_reader.get_mpp(slide_file, highest_level)

        whole_image, meta_data = wsi_reader.get_data(slide_file, level=highest_level)
        height_pixels, width_pixels = meta_data["spatial_shape"]
        width = width_pixels * microns_prpx_x
        height = height_pixels * microns_prpx_y
        n_patches_width = int(width // MIN_PATCH_SIZE)
        n_patches_height = int(height // MIN_PATCH_SIZE)

        assert n_patches_width > 0 and n_patches_height > 0, f"File {file} with patch width of {n_patches_width} and height of {n_patches_height} is too small"

        whole_image_grayscale = whole_image.mean(axis=0)
        resized = cv2.resize(whole_image_grayscale, (n_patches_width, n_patches_height))

        #whole_image_grayscale = (resized < 255 - THRESHOLD * 255).astype(int)
        dataset = output_file.create_dataset(file.as_posix(), data=resized)
        dataset.attrs["width"] = width
        dataset.attrs["height"] = height
        
if __name__ == "__main__":
    main()