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
python prepare_data.py -tr train -ts test -mod wb styled texture orig -o final_cut
```


##### console ...

```
...

Extracting data, creating instance

Creating label subset for 14 class... finished
Creating label subset for 281 class... finished
Creating label subset for 340 class... finished
Creating label subset for 385 class... finished
Creating label subset for 444 class... finished
Creating label subset for 559 class... finished
Creating label subset for 779 class... finished
Creating label subset for 784 class... finished
Creating label subset for 859 class... finished
Creating label subset for 968 class... finished

Start calculating all influences

Preprocessing data... finished


Calculating influence for styled data, label: 14

Test image #1 in process
Calculating gradients for test image... finished
IHVP calculation | 01:28 est. | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 10/10 [01:28<00:00,  8.82s/it]
Inf. calculation | 00:07 est. | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 9/9 [00:07<00:00,  1.17it/s]
Top 5 harmful		->	14/harmful_styled_1.pdf
Top 5 useful		->	14/useful_styled_1.pdf
All influences dict	->	14/styled_1.npy

Test image #2 in process
Calculating gradients for test image... finished
IHVP calculation | 01:32 est. | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 10/10 [01:32<00:00,  9.28s/it]
Inf. calculation | 00:07 est. | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 9/9 [00:07<00:00,  1.22it/s]
Top 5 harmful		->	14/harmful_styled_2.pdf
Top 5 useful		->	14/useful_styled_2.pdf
All influences dict	->	14/styled_2.npy

...
```
