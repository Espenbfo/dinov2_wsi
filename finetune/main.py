import torch
from tqdm import tqdm

from dataset import load_datasets, load_dataloader
from model import init_model, load_model
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.functional import cross_entropy
import json
import time
import numpy as np
from pathlib import Path

DEVICE = "cuda"
EPOCHS = 5
CONTINUE_TRAINING = True
LOSS_MEMORY = 1000 # batches
BATCH_SIZE = 16
CHECKPOINT_TIME = 20 # Minutes
LEARNING_RATE_CLASSIFIER = 0.0001
LEARNING_RATE_FEATURES = 5e-6
FILENAME = "weights.pt"
TRAIN_TRANSFORMER = False
DATASET_FOLDER = Path(r"")
TRAIN_DATASET_FRACTION = 0.95
GRAD_CLIP_VALUE = 0.0
STEPS_PR_SCHEDULER_UPDATE = 1000
SCHEDULER_GAMMA = 0.90
def main():
    print("Cuda available?", torch.cuda.is_available())
    dataset_train, dataset_val, classes = load_datasets(DATASET_FOLDER, train_fraction=TRAIN_DATASET_FRACTION)
    dataloader_train = load_dataloader(dataset_train, BATCH_SIZE, classes,True)
    dataloader_val = load_dataloader(dataset_val, BATCH_SIZE, classes, False)
    if CONTINUE_TRAINING:
        model = load_model(len(classes), "weights.pt").to(DEVICE)
    else:
        model = init_model(len(classes)).to(DEVICE)
    model.transformer.eval()
    params = [{"params": model.classifier.parameters(), "lr": LEARNING_RATE_CLASSIFIER}]

    if TRAIN_TRANSFORMER:
        model.transformer.train()
        params.append({"params": model.transformer.parameters(), "lr": LEARNING_RATE_FEATURES})
    else:
        for parameter in model.transformer.parameters():
            parameter.requires_grad = False
    optimizer_classifier = Adam(params)
    scheduler = ExponentialLR(optimizer_classifier, gamma=0.9)
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
            learning_rates = scheduler.get_lr()
            pbar.postfix = f"mean loss the last {min(index+1, LOSS_MEMORY)} batches {loss:.3f} | accuracy {accuracy:.3f} | time since checkpoint {time.time()-checkpoint_time:.1f}s | Learning rate {learning_rates[0]:.2g}"
            if (time.time() > checkpoint_time+CHECKPOINT_TIME*60):
                torch.save(model.state_dict(), FILENAME)
                checkpoint_time = time.time()
            if ((index+1)%STEPS_PR_SCHEDULER_UPDATE == 0):
                scheduler.step()

        if not TRAIN_TRANSFORMER:
            model.classifier.eval()
        else:
            model.eval()
        print("VALIDATION")
        val_accuracy = 0
        val_loss = 0
        for index, (batch,label) in (pbar := tqdm(enumerate(dataloader_train), total=len(dataloader_val))):
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



if __name__ == "__main__":
    main()
