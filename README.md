# InfluenceCalculations
Influence calculations based on tensorflow 2

**Notes**

- There no eigenvalues calculations because calculated eigenvalues just converges (in limit) to scale parameter and it doesn't make sense to calculate them

## `prepare_data.py`
This script make initial data processing. It takes photos from their directories and create archive with labeled arrays by them classes from these photos.
Args: 
```
-tr   :   path to folder with train ulabeled images
-ts   :   path to folder with test labeled images
-mod  :   path to folders (it may be many folders) with modified images, separated by space
-o    :   output filename of archive  
```

Usage:
```
python prepare_data.py -tr train -ts test -mod wb styled texture orig -o final_cut
```

Normal console output
```
Images from train extraction | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 1000/1000 [00:00<00:00, 6578.50it/s]
Images from test extraction | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 10/10 [00:00<?, ?it/s]
Images from orig extraction | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 10/10 [00:00<00:00, 10104.32it/s]
Images from styled extraction | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 10/10 [00:00<00:00, 9899.23it/s]
Images from texture extraction | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 10/10 [00:00<00:00, 10097.02it/s]
Images from wb extraction | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 10/10 [00:00<00:00, 9995.96it/s]
Creating an archive from arrays... finished
Archive created. Uploading array's names:
['randcore_x', 'randcore_y', 'test_x', 'test_y', 'orig_x', 'orig_y', 'styled_x', 'styled_y', 'texture_x', 'texture_y', 'wb_x', 'wb_y']
```

## `calculate.py`
This script takes a preprocessed data from archive (e.g. final_cut.npz) and:
1. Unpack randomcore and test data, from test_y takes all unique labels and choose from extra data subset by choosed label. After it saves them into archives, e.g. `label_number.npz`.
2. Preprocess images and labels. Create instance model (vgg16).
3. Run through all labels, inside every labels run through all extra data (modified images)
4. Save top 5 harmful and useful images by influence and full influence dict (`.npy`) for each test image into `label_number/`

Args: 
```
-z    :   path to compressed .npz file name with prepared data
-it   :   recursion depth for IHVP calculations
-bs   :   batch size for IHVP calculations
-sv   :   output format for matplotlib figures
```

Usage:
```
python calculate.py -z final_cut -it 300 -bs 8 -sv pdf
```

Nornmal console output

```

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

```
