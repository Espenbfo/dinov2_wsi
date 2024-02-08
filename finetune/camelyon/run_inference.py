from dataset import CamyleonDataset
from monai.data import WSIReader
from torchvision import transforms
import numpy as np
import cv2
import torch
import tqdm

transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224), antialias=False)])


def get_patch_at_location(reader, wsi, location, patch_physical_size, resolution, location_is_center=False):
    resolution_at_level_0 = wsi.shape
    mpp_at_zero = max(reader.get_mpp(wsi, level=0))
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
    for level in range(reader.get_level_count(wsi)):
        mpp = max(reader.get_mpp(wsi, level=level))
        if mpp > target_mpp:
            break
        current_level = level
        mpp_at_level = mpp
    patch_resolution = int(patch_physical_size / mpp_at_level)

    image_before, meta_data = reader.get_data(
        wsi,
        location_coord,
        size=(patch_resolution, patch_resolution),
        level=current_level,
    )
    image = cv2.resize(np.moveaxis(image_before, 0, 2), (resolution, resolution), interpolation=cv2.INTER_AREA)

    return image


def infer_slide(slide_path, batch_size, model, distance_per_sample=100, sizes=(100, 100)):
    reader = WSIReader()
    slide_file = reader.read(slide_path)
    size = np.array(reader.get_size(slide_file, level=0))
    microns_per_pixel = np.array(reader.get_mpp(slide_file, level=0))
    physical_size = size * microns_per_pixel
    n_samples = (physical_size // distance_per_sample).astype(int)

    distance_between_samples = physical_size / distance_per_sample
    rest_size = physical_size % distance_per_sample

    predictions_arr = np.zeros(n_samples)
    ignored_arr = np.zeros(n_samples)

    batch = [[] for _ in sizes]
    batch_indexes = []

    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        # print(physical_size, n_samples, distance_between_samples, rest_size)
        with tqdm.tqdm(total=n_samples[0] * n_samples[1]) as pbar:
            for i in range(n_samples[0]):
                for j in range(n_samples[1]):
                    index = np.array([i, j])
                    coords = (rest_size / 2 + distance_per_sample * index) / physical_size
                    images = [get_patch_at_location(reader, slide_file, coords, size, 224, True) for size in sizes]
                    images = [transforms(image).to("cuda") for image in images]
                    std = torch.std(images[0])
                    if std < 0.02:
                        ignored_arr[i, j] = 1
                        predictions_arr[i, j] = 0
                        pbar.update(1)
                        continue
                    else:
                        for h, image in enumerate(images):
                            batch[h].append(image)
                        batch_indexes.append((i,j))
                        if len(batch[0])==batch_size:
                            batch = (torch.stack(sub_batch) for sub_batch in batch)

                            pred = model(*batch)
                            pred = softmax(pred)
                            pred = pred.detach().cpu()
                            #label = torch.argmax(pred[0])
                            labels = 0.5 + pred[:,1]*0.5 - pred[:,0]*0.5
                            for indexes, label in zip(batch_indexes, labels):
                                predictions_arr[indexes] = label
                            batch = [[] for _ in sizes]
                            batch_indexes = []
                            pbar.update(batch_size)
                    pbar.set_postfix_str(f"std: {std:.3f}")
    return predictions_arr, ignored_arr


def remove_noise(mask, kernel_size=3):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)


def main():
    dataset = CamyleonDataset()
