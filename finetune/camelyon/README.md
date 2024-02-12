## Camelyon dataset
The camyleon dataset is a breast cancer WSI dataset, consisting of 500 annotated slides for training, and 500 slides for testing. https://camelyon17.grand-challenge.org/Home/

### Download
To download parts of or the complete dataset, install aws cli, and run:
```
aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON17/ . --recursive --exclude "*" --include "SOME FILTER"
```
This will only download files matching the `--include` filter. You could for instance set this to `"*004*"` to download all files related to patient 4.

To download all files, run:
```
aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON17/ . --recursive
```

#### Download all files with masks
1. First, download all masks with the command
To download all files, run:
```
aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON17/ . --recursive --exclude "*" --include "masks/*"
```

2. Then run the `find_all_masked_files.ipynb` notebook to get all the filenames in filenames.txt.

3. Then, move the filenames.txt file to the parent folder of the masks folder.

4. Finally, run the bash script below in the same folder as the the filenames.txt file to download all slide images.
Bash script
```
#!/bin/bash  
set -e  
while read line  
do  
  aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON17/$line ./images  
done <filenames.txt && sleep 1
```
### Visualization
You can view the WSIs and annotations in [ASAP](https://github.com/computationalpathologygroup/ASAP)

### Python

