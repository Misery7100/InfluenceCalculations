# InfluenceCalculations
Influence calculations based on tensorflow 2

## `prepare_data.py`
This script make initial data processing. It takes photos from their directories and create archive with labeled arrays by them classes from these photos.
Args: 
```
-tr:  path to folder with train ulabeled images
-ts:  path to folder with test labeled images
-mod: path to folders (it may be many folders) with modified images, separated by space
-o:   output filename of archive  
```

Usage:
```
python prepare_data.py(-tr train -ts test -mod wb styled texture orig -o final_cut)
```
