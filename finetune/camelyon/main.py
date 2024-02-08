import torch
from tqdm import tqdm

from .model import init_model
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import cross_entropy
import json
import time
import numpy as np
from pathlib import Path
import h5py
from .dataset import CamyleonDataset

DEVICE = "cuda"
EPOCHS = 1*8
CONTINUE_TRAINING = False
LOSS_MEMORY = 1000  # batches
BATCH_SIZE = 128
CHECKPOINT_TIME = 20  # Minutes
LEARNING_RATE_CLASSIFIER = 1e-3
LEARNING_RATE_FEATURES = 1e-3
TRAIN_TRANSFORMER = False
STEPS_PR_SCHEDULER_UPDATE = 1000
SCHEDULER_STEPS_PER_EPOCH = 10
SIZES = (64, 224)
EPOCH_MULTIPLIER=64

CHECKPOINT_PATH = Path(
    "weights/teacher_checkpoint-7.pth"
)  # Path("weights/teacher_checkpoint-3.pth")#Path("/home/espenbfo/results/model_0037499.rank_0.pth")
PREPROCESSED_DATASET_PATH = Path("/home/espenbfo/Documents/projects/dinov2_wsi/camelyon.hdf5")
FILENAME = "weights64224-2.pt"


def load_dataloader(dataset, batch_size, classes, shuffle=True):
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle, num_workers=6)


def main():
    print("Cuda available?", torch.cuda.is_available())

    classes = (1, 2)
    dataset_train = CamyleonDataset(PREPROCESSED_DATASET_PATH, sizes=SIZES, iterations_per_epoch_multiplier=EPOCH_MULTIPLIER)
    dataset_val = CamyleonDataset(PREPROCESSED_DATASET_PATH,  sizes=SIZES, is_train=False, iterations_per_epoch_multiplier=EPOCH_MULTIPLIER)

    train_files = dataset_train.files["images"]
    val_files = dataset_val.files["images"]
    assert True not in [file in train_files for file in val_files] # assert that there is no overlap between train and val
    dataloader_train = load_dataloader(dataset_train, BATCH_SIZE, classes, True)
    dataloader_val = load_dataloader(dataset_val, BATCH_SIZE, classes, False)

    model = init_model(len(classes), CHECKPOINT_PATH, teacher_checkpoint=True, n_sizes = len(SIZES)).to(DEVICE)

    params = [{"params": model.classifier.parameters(), "lr": LEARNING_RATE_CLASSIFIER}]

    if TRAIN_TRANSFORMER:
        model.transformer.train()
        params.append({"params": model.transformer.parameters(), "lr": LEARNING_RATE_FEATURES})
    else:
        for parameter in model.transformer.parameters():
            parameter.requires_grad = False

    optimizer_classifier = Adam(params)
    scheduler = CosineAnnealingLR(optimizer_classifier, T_max=EPOCHS * SCHEDULER_STEPS_PER_EPOCH)

    loss_arr = np.zeros(LOSS_MEMORY)
    acc_arr = np.zeros(LOSS_MEMORY)
    checkpoint_time = time.time()
    for epoch in range(EPOCHS):
        print("epoch", epoch + 1)
        total_loss = 0
        print("TRAIN")
        for index, (batch, label) in (pbar := tqdm(enumerate(dataloader_train), total=len(dataloader_train), dynamic_ncols=True)):
            optimizer_classifier.zero_grad()
            batch = (x.to(DEVICE) for x in batch)
            label = label.to(DEVICE)
            result = model(*batch)
            label -= 1
            loss = cross_entropy(result, label)
            loss.backward()

            optimizer_classifier.step()
            loss_arr = np.roll(loss_arr, -1)
            loss_arr[-1] = loss.detach().cpu()
            batch_size = label.size()[0]

            accuracy = torch.eq(label, torch.argmax(result, dim=1)).sum() / batch_size

            acc_arr = np.roll(acc_arr, -1)
            acc_arr[-1] = accuracy
            loss = loss_arr[max(LOSS_MEMORY - index - 1, 0) : LOSS_MEMORY].mean()
            accuracy = acc_arr[max(LOSS_MEMORY - index - 1, 0) : LOSS_MEMORY].mean()
            learning_rates = scheduler.get_last_lr()
            pbar.postfix = f"mean loss the last {min(index+1, LOSS_MEMORY)} batches {loss:.3f} | accuracy {accuracy:.3f} | time since checkpoint {time.time()-checkpoint_time:.1f}s | Learning rate {learning_rates[0]:.2g}"
            if time.time() > checkpoint_time + CHECKPOINT_TIME * 60:
                torch.save(model.state_dict(), FILENAME)
                checkpoint_time = time.time()
            if (index + 1) % (len(dataloader_train) // SCHEDULER_STEPS_PER_EPOCH) == 0:
                scheduler.step()

        if not TRAIN_TRANSFORMER:
            model.classifier.eval()
        else:
            model.eval()
        print("VALIDATION")
        val_accuracy = 0
        val_loss = 0
        with torch.no_grad():
            for index, (batch, label) in (pbar := tqdm(enumerate(dataloader_val), total=len(dataloader_val))):
                batch = (x.to(DEVICE) for x in batch)
                label = label.to(DEVICE)
                label -= 1
                result = model(*batch)
                batch_size = label.size()[0]

                accuracy = torch.eq(label, torch.argmax(result, dim=1)).sum() / batch_size
                loss = cross_entropy(result, label)

                val_accuracy += accuracy
                val_loss += loss.detach().cpu()

        print(
            f"Average batch loss: {val_loss/len(dataloader_val):.3f}, Average batch accuracy {val_accuracy/len(dataloader_val):.3f}"
        )
        dataset_val.reset_rng()
        if not TRAIN_TRANSFORMER:
            model.classifier.train()
        else:
            model.train()
    torch.save(model.state_dict(), FILENAME)


if __name__ == "__main__":
    main()
