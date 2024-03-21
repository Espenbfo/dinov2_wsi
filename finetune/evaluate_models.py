from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
from .dataset import load_dataloader

from .model import init_model

DEVICE = "cuda"
DATASET = "wilds" # One of PCam, wilds, crc, crc_no_norm
ENSEMBLE_STRATEGY = "softmax" #"mode" #"mode"
BATCH_SIZE = 32

model_args = [
    ("normal", "weights_wilds_74999.pt"),
    ("normal", "weights_wilds_87499.pt")
] # Format: (model mode, filename)

match DATASET:
    case "PCam":
        from .PCam import get_pcam_datasets
    case "wilds":
        from .wilds_dataset import get_wilds_datasets    
    case "crc":
        from .crc_dataset import get_crc_datasets
    case "crc_no_norm":
        from .crc_dataset import get_crc_datasets_no_norm

def main():

    match DATASET:
        case "PCam":
            print("PCam")
            dataset_train, dataset_val, dataset_test = get_pcam_datasets()
        case "wilds":
            print("wilds")
            dataset_train, dataset_val, dataset_test = get_wilds_datasets()
        case "crc":
            print("crc")
            dataset_train, dataset_val, dataset_test = get_crc_datasets()
        case "crc_no_norm":
            print("crc_no_norm")
            dataset_train, dataset_val, dataset_test = get_crc_datasets_no_norm()
    classes = dataset_train.classes
        
    dataloader_val = load_dataloader(dataset_val, BATCH_SIZE, classes, False)
    dataloader_test = load_dataloader(dataset_test, BATCH_SIZE, classes, False)

    models = []
    for args in model_args:
        model = init_model(len(classes), None, teacher_checkpoint=True, mode=args[0]).to(DEVICE)
        for parameter in model.parameters():
            parameter.requires_grad = False
        weights = torch.load(args[1])
        model.load_state_dict(weights)
        model.eval()
        models.append(model)

    def run_epoch(dataloader, loss_memory=None, color="white"):
        if loss_memory is None:
            loss_memory = len(dataloader)
        acc_arr = np.zeros(loss_memory)

        individual_accs = [np.zeros(loss_memory) for _ in range(len(models))]
        for index, (batch,label) in (pbar := tqdm(enumerate(dataloader), total=len(dataloader), dynamic_ncols=True, colour=color)):
            batch_size = batch.shape[0]
            batch = batch.to(DEVICE)
            predictions = []
            for model in models:
                predictions.append(model(batch))
            

            for i in range(len(models)):
                individual_accs[i] = np.roll(individual_accs[i],-1)
                individual_accs[i][-1] = torch.eq(label, torch.argmax(predictions[i], dim=1).cpu()).sum()/batch_size

            match ENSEMBLE_STRATEGY:
                case "mode":
                    results = []
                    for prediction in predictions:
                        results.append(torch.argmax(prediction, dim=1).cpu().numpy())
                    ensemble_predictions = torch.zeros((batch_size, ))
                    for i in range(batch_size):
                        preds = np.array([results[j][i] for j in range(len(models))])
                        uniques, counts = np.unique(preds, return_counts=True)
                        max_indexes = np.argwhere(counts==counts.max()).flatten()
                        random_index= np.random.choice(max_indexes)
                        ensemble_predictions[i] = uniques[random_index]

                case "mean":
                    ensemble_predictions = torch.argmax(sum(predictions), dim=1).cpu()

                case "softmax":
                    for i in range(len(predictions)):
                        predictions[i] = torch.nn.functional.softmax(predictions[i], dim=1)
                    ensemble_predictions = torch.argmax(sum(predictions), dim=1).cpu()
                
                case _:
                    print("NO MATCH")
                    assert False


            accuracy = torch.eq(label, ensemble_predictions).sum()/batch_size

            
            acc_arr = np.roll(acc_arr, -1)
            acc_arr[-1] = accuracy
            accuracy = acc_arr[max(loss_memory-index-1,0):loss_memory].mean()

            ind_accs = ", ".join([f"{ind_accuracy[max(loss_memory-index-1,0):loss_memory].mean():.3f}" for ind_accuracy in individual_accs])
            pbar.postfix = f"mean accuracy {accuracy:.3f}, Individual accuracies: {ind_accs}"
        
        final_accuracy = acc_arr.mean()

        return final_accuracy
    
    val_acc = run_epoch(dataloader_val)
    test_acc = run_epoch(dataloader_test)

    print(f"validation accuracy: {val_acc:.1%}, test accuracy: {test_acc:.1%}")