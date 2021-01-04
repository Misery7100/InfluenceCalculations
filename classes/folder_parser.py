# Functions:

# 1. createDict(DATA_DIR)
# It create a dictionary with folder names as keys and image class labels as values.
# Also it writes dictionary on local disk.
# Return: label's dictionary

# 2. stochasticDataSet(DATA_DIR, target_size=(224,224), subs_size=2000, seed = 282, dictionary)
# It create a subset from DNN train data and process this subset for the following actions.
# Return: array of processed images, array of class labels for this images

# 3. imagePreprocessing(DATA_DIR, target_size=(224,224))
# This function make image processing with images in folder, that named as classes of these images.
# Work with pointed images!
# Return: array of processed images, array of class labels for this images

# 3. createZip()
# This is main function: it create necessary archive with images.
# Return: names of files from archive

import csv
import os
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from tqdm import tqdm
import random

class FolderParser:

    def __init__(self, args):
        self.trainDir = args.train
        self.testDir = args.test
        self.modImgList = args.modifiedImages
        self.outName = args.outName

    def createDict(self, DATA_DIR):
        list_dir = []
        dir_dict = {}

        for root, subFolder, files in os.walk(DATA_DIR):
            list_dir.append(subFolder)

        for i in range(len(list_dir[0])):
            dir_dict.update({list_dir[0][i]: str(i)})

        # save our dictionary to labelDict.csv in .csv format

        with open('labelsDict.csv', 'w') as f:
            w = csv.DictWriter(f, dir_dict.keys())
            w.writeheader()
            w.writerow(dir_dict)

    def stochasticDataSet(self, DATA_DIR, target_size=(224, 224), subs_size=2000, seed=282, dictionary=None):
        list_dir = []
        dir_dict = dictionary
        img_subs_path = []
        img_subs_arr = []
        img_subs_lab = []

        for root, subFolder, files in os.walk(DATA_DIR):
            list_dir.append(subFolder)

        for directory in tqdm(list_dir[0], desc='Images from ' + str(DATA_DIR) + ' extraction', ncols=100, ascii=' ▮',
                                bar_format="{desc} | {percentage:.0f}% |{bar}{r_bar}"):
        
            IMG_DIR = os.path.join(DATA_DIR, directory)
            image_names = os.listdir(IMG_DIR)
            for img in image_names:
                image_path = os.path.join(IMG_DIR, img)
                img_subs_path.append((image_path, directory))

        random.seed(seed)
        random.shuffle(img_subs_path)
        img_subs_path = img_subs_path[:subs_size]

        for ind in range(len(img_subs_path)):
            image = load_img(img_subs_path[ind][0], target_size=target_size)
            image = img_to_array(image, dtype=np.int16)
            img_subs_arr.append(image)
            img_subs_lab.append(dir_dict.get(img_subs_path[ind][1]))

        return (np.array(img_subs_arr), np.array(img_subs_lab).astype(np.int32).reshape(-1, 1))

    def imagePreprocessing(self, DATA_DIR, target_size=(224, 224)):
        list_dir = []
        img_subs_path = []
        img_subs_arr = []
        img_subs_lab = []

        for root, subFolder, files in os.walk(DATA_DIR):
            list_dir.append(subFolder)

        for directory in tqdm(list_dir[0], desc='Images from ' + str(DATA_DIR) + ' extraction', ncols=100, ascii=' ▮',
                                bar_format="{desc} | {percentage:.0f}% |{bar}{r_bar}"):
            IMG_DIR = os.path.join(DATA_DIR, directory)
            image_names = os.listdir(IMG_DIR)
            for img in image_names:
                image_path = os.path.join(IMG_DIR, img)
                img_subs_path.append((image_path, directory))

        for ind in range(len(img_subs_path)):
            image = load_img(img_subs_path[ind][0], target_size=target_size)
            image = img_to_array(image, dtype=np.int16)
            img_subs_arr.append(image)
            img_subs_lab.append(img_subs_path[ind][1])

        return (np.array(img_subs_arr), np.array(img_subs_lab).astype(np.int32).reshape(-1, 1))

    def createZip(self):

        # create dictionary from folder's names in "\train" folder
        self.createDict(self.trainDir)
        # read out dictionary from disk
        reader = csv.DictReader(open('labelsDict.csv'))
        labelDictionary = {}
        for row in reader:
            labelDictionary.update(row)

        # create arrays from images and class labels for these images
        # in "\train" folder we have unlabeled images. Let's use for them stochasticDataSet() function:

        randcore_x, randcore_y = self.stochasticDataSet(self.trainDir, (224, 224), 2000, 282, labelDictionary)

        # in "\test" folder and folders with modified images already have class labels, so we use for them
        # imagePreprocessing() function:

        test_x, test_y = self.imagePreprocessing(self.testDir, (224, 224))

        exec_dir = []
        for directory in self.modImgList:
            exec(f'{directory}_x, {directory}_y, = self.imagePreprocessing("{directory}", (224, 224))')
            exec_dir.append(f'{directory}_x={directory}_x')
            exec_dir.append(f'{directory}_y={directory}_y')

        exec_dir = ','.join(exec_dir)

        print('Creating an archive from arrays... ', end='')

        exec(f'np.savez("{self.outName}",\
                 randcore_x=randcore_x, randcore_y=randcore_y,\
                 test_x=test_x, test_y=test_y,{exec_dir})')

        print('finished')
        print('Archive created. Uploading array\'s names:')

        del randcore_x, randcore_y

        ld = np.load(f'{self.outName}.npz')
        files = ld.files

        del ld

        return files