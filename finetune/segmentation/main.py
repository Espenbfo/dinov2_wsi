import torch
from tqdm import tqdm

from .model import init_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import cross_entropy
import json
import time
import numpy as np
from pathlib import Path
import h5py
from .dataset import CamyleonDataset
torch.manual_seed(12)

DEVICE = "cuda"
EPOCHS = 10
CONTINUE_TRAINING = False
LOSS_MEMORY = 1000  # batches
BATCH_SIZE = 32
CHECKPOINT_TIME = 20  # Minutes
LEARNING_RATE_CLASSIFIER = 1e-4
LEARNING_RATE_FEATURES = 1e-3
TRAIN_TRANSFORMER = False
SCHEDULER_STEPS_PER_EPOCH = 1
EPOCH_MULTIPLIER=160
EPOCH_MULTIPLIER_VAL=1000
EARLY_STOPPING_MEMORY = 20
PREPROCESSED_DATASET_PATH = Path("/home/espenbfo/Documents/projects/dinov2_wsi/camelyon.hdf5")
CLASS_WEIGHTS_TRAIN = (0.90, 0.10)
SIZES_AND_BACKBONES = (
    (96, "phikon", None),
    (288, "normal", "weights/a100_full_87499.pth")
    )

FILENAME = f"weights{'-'.join(map(lambda x: str(x[0]), SIZES_AND_BACKBONES))}.pt"


def load_dataloader(dataset, batch_size, classes, shuffle=True):
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle, num_workers=6)


def main():
    print("Cuda available?", torch.cuda.is_available())

    classes = (1, 2)
    sizes = [x[0] for x in SIZES_AND_BACKBONES]
    dataset_train = CamyleonDataset(PREPROCESSED_DATASET_PATH, sizes=sizes, iterations_per_epoch_multiplier=EPOCH_MULTIPLIER, class_weights=CLASS_WEIGHTS_TRAIN)
    dataset_val = CamyleonDataset(PREPROCESSED_DATASET_PATH,  sizes=sizes, is_train=False, iterations_per_epoch_multiplier=EPOCH_MULTIPLIER_VAL)

    train_files = dataset_train.files["images"]
    val_files = dataset_val.files["images"]
    assert True not in [file in train_files for file in val_files] # assert that there is no overlap between train and val
    dataloader_train = load_dataloader(dataset_train, BATCH_SIZE, classes, True)
    dataloader_val = load_dataloader(dataset_val, BATCH_SIZE, classes, False)

    model = init_model(len(classes), SIZES_AND_BACKBONES).to(DEVICE)

    params = [{"params": model.classifier.parameters(), "lr": LEARNING_RATE_CLASSIFIER}]

    if TRAIN_TRANSFORMER:
        model.transformer.train()
        params.append({"params": model.backbones.parameters(), "lr": LEARNING_RATE_FEATURES})
    else:
        for parameter in model.backbones.parameters():
            parameter.requires_grad = False

    optimizer_classifier = AdamW(params)
    scheduler = CosineAnnealingLR(optimizer_classifier, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    loss_arr = np.zeros(LOSS_MEMORY)
    acc_arr = np.zeros(LOSS_MEMORY)
    checkpoint_time = time.time()
    best_val_loss = 100000
    epochs_since_improvement = 0


    def run_epoch(dataloader, is_train, loss_memory=None, color="white"):
        if loss_memory is None:
            loss_memory = len(dataloader)
        else:
            loss_memory = min(len(dataloader), loss_memory)
        loss_arr = np.zeros(loss_memory)
        acc_arr = np.zeros(loss_memory)

        loss_weights = (1-torch.tensor(CLASS_WEIGHTS_TRAIN)).to(DEVICE)*len(CLASS_WEIGHTS_TRAIN)
        for index, (batch,label) in (pbar := tqdm(enumerate(dataloader), total=len(dataloader), dynamic_ncols=True, colour=color)):
            with torch.autocast("cuda", torch.float16):
                if is_train:
                    optimizer_classifier.zero_grad()
                batch = [x.to(DEVICE) for x in batch]
                batch_size = batch[0].shape[0]
                label = label.to(DEVICE)
                label -= 1 # Yup, Spaghetti
                result = model(*batch)
                if is_train:
                    loss = cross_entropy(result, label, loss_weights)
                else:
                    loss = cross_entropy(result, label)
                accuracy = torch.eq(label, torch.argmax(result, dim=1)).sum()/batch_size

                if is_train:
                    scaler.scale(loss).backward()

                    scaler.step(optimizer_classifier)
                    scaler.update()

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

    for epoch in range(EPOCHS):
        print("epoch", epoch + 1)
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

        dataset_val.reset_rng()
        if not TRAIN_TRANSFORMER:
            model.classifier.train()
        else:
            model.train()
        if (val_loss > best_val_loss):
            print("Val loss did not improve")
            epochs_since_improvement += 1
            if epochs_since_improvement == EARLY_STOPPING_MEMORY:
                break
        else:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), FILENAME)



if __name__ == "__main__":
    main()
