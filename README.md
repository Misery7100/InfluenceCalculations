# InfluenceCalculations
Influence calculations based on tensorflow 2

![](https://imgur.com/twNmNHM.jpg)
![](https://imgur.com/heNuGVO.jpg)

**Requirements**

- `tensorflow==2.3.1`
- `keras==2.4.3`
- `tqdm==4.50.2`
- `numpy==1.18.5`
- `matplotlib==3.3.1`
- `pillow==8.0.0`
- `h5py==2.10.0`

**Notes**

- There no eigenvalues calculations because calculated eigenvalues just converges (in limit) to scale parameter and it doesn't make sense to calculate them;
- Added sumup gradients calculation to speed up a process;
- Added batch packaging for train data to reduce memory usage in calculation process.

### Data used
- Tiny ImageNET dataset and ILSVRC2012 which are available on official web-site: http://www.image-net.org (authorization required)
- Links to images used in test calculations: < links >

### Preprocessing data for test calculations
1. Styled images: pre-trained model https://github.com/misgod/fast-neural-style-keras
2. Textured image: additive software MATLAB with VGG19 CNN https://www.mathworks.com/help/images/neural-style-transfer-using-deep-learning.html
3. Background removal: manual editing, cause pre-trained model don't show correct results on our images

## Scripts description

### `prepare_data.py`
This script make initial data processing. It takes photos from their directories and create archive with labeled arrays by them classes from these photos.
Args: 
```
-tr   :   path to folder with train labeled images
-ts   :   path to folders with test labeled images (it may be many folders)
-bs   :   batch size for splitting train data
```

Usage:
```console
python prepare_data.py -tr train -ts orig wb texture styled -bs 150
```

Console output
```
... (maybe some tensorflow warnings)

Parse train | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 1000/1000 [00:00<00:00, 1861.83it/s]
Pack batches | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 139/139 [09:09<00:00,  3.96s/it]
Create train archive... finished
Parse orig | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 1/1 [00:00<00:00, 12710.01it/s]
Parse wb | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 1/1 [00:00<00:00, 19418.07it/s]
Parse styled | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 1/1 [00:00<00:00, 19239.93it/s]
Parse texture | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 1/1 [00:00<00:00, 20971.52it/s]
Create test archive... finished
```

### `prepare_only_test.py`
Additional script to avoid train data reprocessing each time.
Args: 

```
-ts   :   path to folders with test labeled images (it may be many folders)
```

Usage:
```console
python prepare_data.py -ts orig wb texture styled
```

Console output
```
... (maybe some tensorflow warnings)

Parse orig | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 1/1 [00:00<00:00, 12710.01it/s]
Parse wb | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 1/1 [00:00<00:00, 19418.07it/s]
Parse styled | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 1/1 [00:00<00:00, 19239.93it/s]
Parse texture | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 1/1 [00:00<00:00, 20971.52it/s]
Create test archive... finished
```

### `calculate.py`
This script takes a preprocessed data from archives `train.npz` and `test.npz` and calculates influences of train data on test data. Detailed:

1. Create instance model (VGG16 in our particular case);
3. Run through all test folders label by label, firstly calculate inverse hessian vector products and after calculate influences of train data with the same label;
4. Save top 5 harmful and useful images by influence and full influence dict (`.npy`) for each test image into `calc_output/<test_folder>/<label>`

Args: 
```
-i   :   recursion depth for IHVP calculations
-b   :   batch size for IHVP calculations
-sf   :   output format for matplotlib figures
```

Usage:
```console
python calculate.py -i 20 -bs 8 -sf pdf
```

Console output

```
... (maybe some tensorflow warnings)

Start calculating all influences

Calculating influences for folder: orig
Create label subset for 779 class... finished
Calculating influences for class: 779

Test image #1 in process
Calculating gradients for test image... finished
IHVP calculation | 07:57 est. | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 20/20 [07:57<00:00, 23.85s/it]
Batch influences | 01:56 est. | 100% |▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮| 139/139 [01:56<00:00,  1.19it/s]
Top 5 harmful		->	calc_output/orig/779/harmful_1.pdf
Top 5 useful		->	calc_output/orig/779/useful_1.pdf
All influences dict	->	calc_output/orig/779/1.npy

...

```

## User guide

1. Make sure you have all files downloaded and all requirements installed
2. Check your RAM and CPU/GPU usage, free up as much memory as possible to avoid freezes
3. Make sure you have train folder and test folders
    * train folder should contain 1000 folders (if we consider 1k-class classification task) named according to ImageNet notation (from `n01440764` to `n15075141`)
    * test folders should contain folders named according to mapped classes (from `0` to `999`)

