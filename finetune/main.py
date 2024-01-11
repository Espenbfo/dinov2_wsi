import torch
from tqdm import tqdm

from .dataset import PathologyDataset, load_datasets, load_dataloader
from .model import init_model, load_model
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import cross_entropy
import json
import time
import numpy as np
from pathlib import Path
import h5py

DEVICE = "cuda"
EPOCHS = 1
CONTINUE_TRAINING = False
LOSS_MEMORY = 1000 # batches
BATCH_SIZE = 32
CHECKPOINT_TIME = 20 # Minutes
LEARNING_RATE_CLASSIFIER =1e-4
LEARNING_RATE_FEATURES = 1e-4
FILENAME = "weights.pt"
TRAIN_TRANSFORMER = False
GRAD_CLIP_VALUE = 0.0
STEPS_PR_SCHEDULER_UPDATE = 1000
SCHEDULER_GAMMA = 0.70
TRAIN_X_PATH = Path("/home/espenbfo/datasets/classification/pcam/training_split.h5")
TRAIN_Y_PATH = Path("/home/espenbfo/datasets/classification/Labels/Labels/camelyonpatch_level_2_split_train_y.h5")
TRAIN_X_PATH_VAL = Path("/home/espenbfo/datasets/classification/pcam/validation_split.h5")
TRAIN_Y_PATH_VAL = Path("/home/espenbfo/datasets/classification/Labels/Labels/camelyonpatch_level_2_split_valid_y.h5")
TRAIN_X_PATH_TEST = Path("/home/espenbfo/datasets/classification/pcam/test_split.h5")
TRAIN_Y_PATH_TEST = Path("/home/espenbfo/datasets/classification/Labels/Labels/camelyonpatch_level_2_split_test_y.h5")
CHECKPOINT_PATH = Path("/home/espenbfo/results/model_0178499.rank_0.pth")
def main():
    print("Cuda available?", torch.cuda.is_available())


    fx = h5py.File(TRAIN_X_PATH, "r")
    print(fx.keys())
    train_x = fx["x"]
    fy = h5py.File(TRAIN_Y_PATH, "r")
    print(fy.keys())
    train_y = fy["y"]

    fx_val = h5py.File(TRAIN_X_PATH_VAL, "r")
    train_x_val = fx_val["x"]
    fy_val = h5py.File(TRAIN_Y_PATH_VAL, "r")
    train_y_val = fy_val["y"]


    fx_test = h5py.File(TRAIN_X_PATH_TEST, "r")
    train_x_test = fx_test["x"]
    fy_test = h5py.File(TRAIN_Y_PATH_TEST, "r")
    train_y_test = fy_test["y"]


    # train_y = np.load(TRAIN_Y_PATH, allow_pickle=True)
    print(train_x.shape)
    dataset_train = PathologyDataset(train_x, train_y)
    dataset_val = PathologyDataset(train_x_val, train_y_val)
    dataset_test = PathologyDataset(train_x_test, train_y_test)
    classes = dataset_train.classes

    # dataset_train, dataset_val, classes = load_datasets(DATASET_FOLDER, train_fraction=TRAIN_DATASET_FRACTION)
    dataloader_train = load_dataloader(dataset_train, BATCH_SIZE, classes,True)
    dataloader_val = load_dataloader(dataset_val, BATCH_SIZE, classes, False)
    dataloader_test = load_dataloader(dataset_test, BATCH_SIZE, classes, False)

    if CONTINUE_TRAINING:
        model = load_model(len(classes), "weights.pt").to(DEVICE)
    else:
        model = init_model(len(classes), CHECKPOINT_PATH).to(DEVICE)
    model.transformer.eval()
    params = [{"params": model.classifier.parameters(), "lr": LEARNING_RATE_CLASSIFIER}]

    if TRAIN_TRANSFORMER:
        model.transformer.train()
        params.append({"params": model.transformer.parameters(), "lr": LEARNING_RATE_FEATURES})
    else:
        for parameter in model.transformer.parameters():
            parameter.requires_grad = False
    optimizer_classifier = Adam(params)
    scheduler = CosineAnnealingLR(optimizer_classifier, T_max=EPOCHS)
    with open("classes.json", "w") as f:
        json.dump(classes, f)

    loss_arr = np.zeros(LOSS_MEMORY)
    acc_arr = np.zeros(LOSS_MEMORY)
    checkpoint_time = time.time()
    for epoch in range(EPOCHS):
        print("epoch", epoch+1)
        total_loss = 0
        print("TRAIN")
        for index, (batch,label) in (pbar := tqdm(enumerate(dataloader_train), total=len(dataloader_train))):
            optimizer_classifier.zero_grad()
            batch = batch.to(DEVICE)
            label = label.to(DEVICE)
            result = model(batch)
            loss = cross_entropy(result, label)
            loss.backward()

            if (GRAD_CLIP_VALUE != 0.0):
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               GRAD_CLIP_VALUE)

            optimizer_classifier.step()
            loss_arr = np.roll(loss_arr, -1)
            loss_arr[-1] = loss.detach().cpu()
            accuracy = torch.eq(label, torch.argmax(result, dim=1)).sum()/BATCH_SIZE

            acc_arr = np.roll(acc_arr, -1)
            acc_arr[-1] = accuracy
            loss = loss_arr[max(LOSS_MEMORY-index-1,0):LOSS_MEMORY].mean()
            accuracy = acc_arr[max(LOSS_MEMORY-index-1,0):LOSS_MEMORY].mean()
            learning_rates = scheduler.get_last_lr()
            pbar.postfix = f"mean loss the last {min(index+1, LOSS_MEMORY)} batches {loss:.3f} | accuracy {accuracy:.3f} | time since checkpoint {time.time()-checkpoint_time:.1f}s | Learning rate {learning_rates[0]:.2g}"
            if (time.time() > checkpoint_time+CHECKPOINT_TIME*60):
                torch.save(model.state_dict(), FILENAME)
                checkpoint_time = time.time()
            if ((index+1)%STEPS_PR_SCHEDULER_UPDATE == 0):
                pass
        scheduler.step()

        if not TRAIN_TRANSFORMER:
            model.classifier.eval()
        else:
            model.eval()
        print("VALIDATION")
        val_accuracy = 0
        val_loss = 0
        with torch.no_grad():
            for index, (batch,label) in (pbar := tqdm(enumerate(dataloader_val), total=len(dataloader_val))):
                batch = batch.to(DEVICE)
                label = label.to(DEVICE)

                result = model(batch)
                accuracy = torch.eq(label, torch.argmax(result, dim=1)).sum()/BATCH_SIZE
                loss = cross_entropy(result, label)

                val_accuracy += accuracy
                val_loss += loss.detach().cpu()

        print(f"Average batch loss: {val_loss/len(dataloader_val)}, Average batch accuracy {val_accuracy/len(dataloader_val)}")

        if not TRAIN_TRANSFORMER:
            model.classifier.train()
        else:
            model.train()

    print("TEST")
    test_accuracy = 0
    test_loss = 0
    with torch.no_grad():
        for index, (batch,label) in (pbar := tqdm(enumerate(dataloader_test), total=len(dataloader_test))):
            batch = batch.to(DEVICE)
            label = label.to(DEVICE)

            result = model(batch)
            accuracy = torch.eq(label, torch.argmax(result, dim=1)).sum()/BATCH_SIZE
            loss = cross_entropy(result, label)

            test_accuracy += accuracy
            test_loss += loss.detach().cpu()

    print(f"Average batch loss: {test_loss/len(dataloader_test)}, Average batch accuracy {test_accuracy/len(dataloader_test)}")

if __name__ == "__main__":
    main()
