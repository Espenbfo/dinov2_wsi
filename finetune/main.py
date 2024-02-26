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
EPOCHS = 8
CONTINUE_TRAINING = False
LOSS_MEMORY = 1000 # batches
BATCH_SIZE = 32
LEARNING_RATE_CLASSIFIER =1e-3
LEARNING_RATE_FEATURES = 1e-4
FILENAME = "weights.pt"
TRAIN_TRANSFORMER = False
EARLY_STOPPING_MEMORY = 10
DATASET = "PCam" # One of PCam, wilds

match DATASET:
    case "PCam":
        from .PCam import get_pcam_datasets
    case "wilds":
        from .wilds_dataset import get_wilds_datasets

CHECKPOINT_PATH = Path("weights/teacher_checkpoint-8.pth")#Path("weights/teacher_checkpoint-3.pth")#Path("/home/espenbfo/results/model_0037499.rank_0.pth")
def main():
    print("Cuda available?", torch.cuda.is_available())


    # train_y = np.load(TRAIN_Y_PATH, allow_pickle=True)
    match DATASET:
        case "PCam":
            dataset_train, dataset_val, dataset_test = get_pcam_datasets()
        case "wilds":
            dataset_train, dataset_val, dataset_test = get_wilds_datasets()
    classes = dataset_train.classes


    # dataset_train, dataset_val, classes = load_datasets(DATASET_FOLDER, train_fraction=TRAIN_DATASET_FRACTION)
    dataloader_train = load_dataloader(dataset_train, BATCH_SIZE, classes,True)
    dataloader_val = load_dataloader(dataset_val, BATCH_SIZE, classes, False)
    dataloader_test = load_dataloader(dataset_test, BATCH_SIZE, classes, False)

    if CONTINUE_TRAINING:
        model = load_model(len(classes), "weights.pt").to(DEVICE)
    else:
        model = init_model(len(classes), CHECKPOINT_PATH, teacher_checkpoint=True).to(DEVICE)
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

    def run_epoch(dataloader, is_train, loss_memory=None, color="white"):
        if loss_memory is None:
            loss_memory = len(dataloader)
        loss_arr = np.zeros(loss_memory)
        acc_arr = np.zeros(loss_memory)

        for index, (batch,label) in (pbar := tqdm(enumerate(dataloader), total=len(dataloader), dynamic_ncols=True, colour=color)):
            if is_train:
                optimizer_classifier.zero_grad()
            batch_size = batch.shape[0]
            batch = batch.to(DEVICE)
            label = label.to(DEVICE)
            result = model(batch)
            loss = cross_entropy(result, label)
            accuracy = torch.eq(label, torch.argmax(result, dim=1)).sum()/batch_size

            if is_train:
                loss.backward()

                optimizer_classifier.step()

            loss_arr = np.roll(loss_arr, -1)
            loss_arr[-1] = loss.detach().cpu()
            acc_arr = np.roll(acc_arr, -1)
            acc_arr[-1] = accuracy
            loss = loss_arr[max(loss_memory-index-1,0):loss_memory].mean()
            accuracy = acc_arr[max(loss_memory-index-1,0):loss_memory].mean()

            if is_train:
                learning_rates = scheduler.get_last_lr()
                pbar.postfix = f"mean loss the last {min(index+1, loss_memory)} batches {loss:.3f} | accuracy {accuracy:.3f} | Learning rate {learning_rates[0]:.2g}"
            else:
                pbar.postfix = f"mean loss the last {min(index+1, loss_memory)} batches {loss:.3f} | accuracy {accuracy:.3f}"
        final_loss = loss_arr.mean()
        final_accuracy = acc_arr.mean()

        return final_loss, final_accuracy

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        print("epoch", epoch+1)
        total_loss = 0
        print("TRAIN")
        train_loss, train_accuracy = run_epoch(dataloader_train, is_train=True, loss_memory=LOSS_MEMORY, color="green")
        print(f"Train loss {train_loss:.3f} | train accuracy {train_accuracy:.3f}")
        scheduler.step()
        if not TRAIN_TRANSFORMER:
            model.classifier.eval()
        else:
            model.eval()
        print("VALIDATION")
        val_accuracy = 0
        val_loss = 0
        with torch.no_grad():
            val_loss, val_accuracy = run_epoch(dataloader_val, is_train=False, color="blue")
        print(f"Validation loss {val_loss:.3f} | validation accuracy {val_accuracy:.3f}")

        if (val_loss > best_val_loss):
            print("Val loss did not improve")
            epochs_since_improvement += 1
            if epochs_since_improvement == EARLY_STOPPING_MEMORY:
                break
        else:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), FILENAME)
        
        if not TRAIN_TRANSFORMER:
            model.classifier.train()
        else:
            model.train()

    print("TEST")
    with torch.no_grad():
        test_loss, test_accuracy = run_epoch(dataloader_test, is_train=False, color="red")
    print(f"Test loss {test_loss:.3f} | test accuracy {test_accuracy:.3f}")

if __name__ == "__main__":
    main()
