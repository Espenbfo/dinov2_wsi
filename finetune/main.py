import torch
from tqdm import tqdm

from .dataset import PathologyDataset, choose_dataset, load_dataloader
from .model import init_model, load_model
from torch.optim import Adam, SGD, AdamW
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
LEARNING_RATE_CLASSIFIER =1e-3
LEARNING_RATE_FEATURES = 5e-5
FILENAME = "weights_1_epoch.pt"
TRAIN_TRANSFORMER = not True
EARLY_STOPPING_MEMORY = 15
DATASET = "PCam" # One of PCam, wilds, crc, crc_no_norm, BACH
MODEL_MODE = "normal" # One of "normal", "dino" for dinov2 trained on natural images, or "phikon" for the phikon model
CHECKPOINT_PATH = Path("/home/bgstovel/Downloads/training_499999_vmamba_tiny_long/teacher_checkpoint.pth")#Path("weights/teacher_checkpoint-3.pth")#Path("/home/espenbfo/results/model_0037499.rank_0.pth")
VALIDATION_FREQ = 1
RUNS_PER_CHECKPOINT = 4
JSON_FILENAME="results_constant.json"


def main():
    results = {}
    if CHECKPOINT_PATH.is_dir():
        checkpoint_paths = list(CHECKPOINT_PATH.rglob("**/*.pth"))
    else:
        checkpoint_paths = [CHECKPOINT_PATH]

    print(checkpoint_paths)
    print(f"Running finetuning on {len(checkpoint_paths)} models")
    print("Cuda available?", torch.cuda.is_available())

    for checkpoint_path in checkpoint_paths:
        results[checkpoint_path.as_posix()] = []

        for run in range(RUNS_PER_CHECKPOINT):
            results[checkpoint_path.as_posix()].append({})
            print(f"Running checkpoint {checkpoint_path}")
            dataset_train, dataset_val, dataset_test = choose_dataset(DATASET)

            classes = dataset_train.classes

            # dataset_train, dataset_val, classes = load_datasets(DATASET_FOLDER, train_fraction=TRAIN_DATASET_FRACTION)
            dataloader_train = load_dataloader(dataset_train, BATCH_SIZE, classes,True)
            dataloader_val = load_dataloader(dataset_val, BATCH_SIZE, classes, False)
            dataloader_test = load_dataloader(dataset_test, BATCH_SIZE, classes, False)

            if CONTINUE_TRAINING:
                model = init_model(len(classes), checkpoint_path, teacher_checkpoint=True, mode=MODEL_MODE).to(DEVICE)
                weights = torch.load("weights_full.pt")
                model.load_state_dict(weights)
                print("Continuing Training")
            else:
                model = init_model(len(classes), checkpoint_path, teacher_checkpoint=True, mode=MODEL_MODE).to(DEVICE)
            model.transformer.eval()
            params = [{"params": model.classifier.parameters(), "lr": LEARNING_RATE_CLASSIFIER}]

            if TRAIN_TRANSFORMER:
                model.transformer.train()
                params.append({"params": model.transformer.parameters(), "lr": LEARNING_RATE_FEATURES})
            else:
                for parameter in model.transformer.parameters():
                    parameter.requires_grad = False
            optimizer_classifier = AdamW(params, weight_decay=0.1)
            scaler = torch.cuda.amp.GradScaler()
            scheduler = CosineAnnealingLR(optimizer_classifier, T_max=EPOCHS)
            with open("classes.json", "w") as f:
                json.dump(classes, f)

            def run_epoch(dataloader, is_train, loss_memory=None, color="white"):
                if loss_memory is None:
                    loss_memory = len(dataloader)
                else:
                    loss_memory = min(len(dataloader), loss_memory)
                loss_arr = np.zeros(loss_memory)
                acc_arr = np.zeros(loss_memory)

                for index, (batch,label) in (pbar := tqdm(enumerate(dataloader), total=len(dataloader), dynamic_ncols=True, colour=color)):
                    with torch.autocast("cuda", torch.float16):
                        if is_train:
                            optimizer_classifier.zero_grad()
                        batch_size = batch.shape[0]
                        batch = batch.to(DEVICE)
                        label = label.to(DEVICE)
                        result = model(batch)
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

            best_val_loss = float("inf")
            for epoch in range(EPOCHS):
                print("epoch", epoch+1)
                total_loss = 0
                print("TRAIN")
                train_loss, train_accuracy = run_epoch(dataloader_train, is_train=True, loss_memory=LOSS_MEMORY, color="green")
                print(f"Train loss {train_loss:.3f} | train accuracy {train_accuracy:.3f}")
                scheduler.step()

                if ((epoch+1)%VALIDATION_FREQ==0):
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
                        results[checkpoint_path.as_posix()][run]["Train Loss"] = train_loss
                        results[checkpoint_path.as_posix()][run]["Train Accuracy"] = train_accuracy
                        results[checkpoint_path.as_posix()][run]["Val Loss"] = val_loss
                        results[checkpoint_path.as_posix()][run]["Val Accuracy"] = val_accuracy
                        best_val_loss = val_loss
                        epochs_since_improvement = 0
                        torch.save(model.state_dict(), FILENAME)

                    if not TRAIN_TRANSFORMER:
                        model.classifier.train()
                    else:
                        model.train()

            if not TRAIN_TRANSFORMER:
                model.classifier.eval()
            else:
                model.eval()
            print("TEST")
            with torch.no_grad():
                state_dict = torch.load(FILENAME)
                model.load_state_dict(state_dict)
                test_loss, test_accuracy = run_epoch(dataloader_test, is_train=False, color="red")
                results[checkpoint_path.as_posix()][run]["Test Loss"] = test_loss
                results[checkpoint_path.as_posix()][run]["Test Accuracy"] = test_accuracy

            print(f"Test loss {test_loss:.3f} | test accuracy {test_accuracy:.3f}")
            with open(JSON_FILENAME, "w") as f:
                json.dump(results, f)
if __name__ == "__main__":
    main()
