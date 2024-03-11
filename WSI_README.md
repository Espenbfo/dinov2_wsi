### Preprocessing command

```
python dinov2/data/datasets/preprocess_wsi_thumbnails.py DATASET_FOLDER thumbnails.hdf5
```
We have observed a speed up of the thumbnail sampling by a factor of 300-400 when using thumbnail preprocessing, which leads to a total speedup of around 5x when sampling a random slide area



## Preprocessing Camelyon for segmentation
```
python finetune/camelyon/preprocess.py DATASET_FOLDER/images DATASET_FOLDER/masks camelyon.hdf5
```


## Finetuning:
Edit finetune/main to your preffered parameters, and then run
```bash
python finetune.py
```
If you want to use the phikon backbone, you need to first add the HistoSSLscaling repository as a submodule:
```bash
git submodule add https://github.com/owkin/HistoSSLscaling.git
```
Install the missing requirements (loguru, einops)
And download the model weights. (You can find the weights in the readme)

