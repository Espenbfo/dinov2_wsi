from pathlib import Path
import h5py

from .dataset import PathologyDataset
PCAM_BASE_PATH = Path("/home/espenbfo/datasets/classification")

TRAIN_X_PATH = PCAM_BASE_PATH / "pcam/training_split.h5"
TRAIN_Y_PATH = PCAM_BASE_PATH / "Labels/Labels/camelyonpatch_level_2_split_train_y.h5"
TRAIN_X_PATH_VAL = PCAM_BASE_PATH / "pcam/validation_split.h5"
TRAIN_Y_PATH_VAL = PCAM_BASE_PATH / "Labels/Labels/camelyonpatch_level_2_split_valid_y.h5"
TRAIN_X_PATH_TEST = PCAM_BASE_PATH / "pcam/test_split.h5"
TRAIN_Y_PATH_TEST = PCAM_BASE_PATH / "Labels/Labels/camelyonpatch_level_2_split_test_y.h5"

def load_dataset_from_h5(h5_path_x, h5_path_y):
    fx = h5py.File(h5_path_x, "r")
    train_x = fx["x"]

    fy = h5py.File(h5_path_y, "r")
    train_y = fy["y"]

    return PathologyDataset(train_x, train_y)

def get_pcam_datasets():
    train, val, test = load_dataset_from_h5(TRAIN_X_PATH, TRAIN_Y_PATH), load_dataset_from_h5(TRAIN_X_PATH_VAL, TRAIN_Y_PATH_VAL), load_dataset_from_h5(TRAIN_X_PATH_VAL, TRAIN_Y_PATH_VAL)

    return train, val, test