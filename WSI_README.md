### Preprocessing command

```
python dinov2/data/datasets/preprocess_wsi_thumbnails.py DATASET_FOLDER thumbnails.hdf5
```
We have observed a speed up of the thumbnail sampling by a factor of 300-400 when using thumbnail preprocessing, which leads to a total speedup of around 5x when sampling a random slide area



## Preprocessing Camelyon for segmentation
```
python finetune/camelyon/preprocess.py DATASET_FOLDER/images DATASET_FOLDER/masks camelyon.hdf5
```