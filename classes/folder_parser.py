import csv
import os
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from tqdm import tqdm
import math
import random

class FolderParser:

    def __init__(self, args):
        try:
            self.trainDir = args.train
        except:
            self.trainDir = None

        self.testDirList = args.test

        try:
            self.batchSize = args.batchSize
        except:
            self.batchSize = None

    def createDict(self):

        dir_dict = {}

        list_dir = list(sorted(os.listdir(self.trainDir)))

        for i, dir in enumerate(list_dir):
            dir_dict[dir] = i


        np.save('labelsDict.npy', dir_dict)


    def batch(self, iterable):
        num = self.batchSize
        length = len(iterable)
        out = []
        for ndx in range(0, length, num):
             out.append(iterable[ndx:min(ndx + num, length)])
        
        return out

    def createTrain(self, target_size=(224, 224), seed=282, dictionary=None):

        dir_dict = dictionary
        img_subs_path = []

        list_dir = list(sorted(os.listdir(self.trainDir)))

        for directory in tqdm(list_dir, desc=f'Parse {self.trainDir}', ncols=100, ascii=' ▮',
                                bar_format="{desc} | {percentage:.0f}% |{bar}{r_bar}"):
        
            IMG_DIR = os.path.join(self.trainDir, directory)
            image_names = os.listdir(IMG_DIR)
            for img in image_names:
                image_path = os.path.join(IMG_DIR, img)
                img_subs_path.append((image_path, directory))

        random.seed(seed)
        random.shuffle(img_subs_path)

        numIter = int(math.ceil(len(img_subs_path)/self.batchSize))

        exec_dir = []
        iterbl = self.batch(img_subs_path)
        i = 0
        for sbs in tqdm(iterbl, desc=f'Pack batches', ncols=100, ascii=' ▮', total=numIter,
                                bar_format="{desc} | {percentage:.0f}% |{bar}{r_bar}"):
            img_subs_arr = []
            img_subs_lab = []
            img_subs = sbs
            for ind in range(len(img_subs)):
                image = load_img(img_subs[ind][0], target_size=target_size)
                image = img_to_array(image, dtype=np.int16)
                img_subs_arr.append(image)
                img_subs_lab.append(dir_dict.get(img_subs[ind][1]))

            exec(f'train_x_{i}, train_y_{i}, = img_subs_arr, np.array(img_subs_lab).reshape(-1,1)')
            exec_dir.append(f'train_x_{i}=train_x_{i}')
            exec_dir.append(f'train_y_{i}=train_y_{i}')

            i += 1

        print('Create train archive... ', end='')

        exec_dir = ','.join(exec_dir)

        exec(f'np.savez("train", {exec_dir})')

        print('finished')

    def createTest(self, target_size=(224, 224)):
        list_dir = self.testDirList
        img_subs_path = []
        exec_dir = []

        # i = 0
        for folder in list_dir:

            img_subs_arr = []
            img_subs_lab = []
            img_subs_path = []

            list_subDir = os.listdir(folder)


            for directory in tqdm(list_subDir, desc=f'Parse {folder}', ncols=100, ascii=' ▮',
                                bar_format="{desc} | {percentage:.0f}% |{bar}{r_bar}"):
                IMG_DIR = os.path.join(folder, directory)
                image_names = os.listdir(IMG_DIR)
                for img in image_names:
                    image_path = os.path.join(IMG_DIR, img)
                    img_subs_path.append((image_path, directory))

            for ind in range(len(img_subs_path)):
                image = load_img(img_subs_path[ind][0], target_size=target_size)
                image = img_to_array(image, dtype=np.int16)
                img_subs_arr.append(image)
                img_subs_lab.append(int(img_subs_path[ind][1]))

            exec(f'test_{folder}_x, test_{folder}_y, = img_subs_arr, np.array(img_subs_lab).reshape(-1,1)')
            exec_dir.append(f'test_{folder}_x=test_{folder}_x')
            exec_dir.append(f'test_{folder}_y=test_{folder}_y')

        exec_dir = ','.join(exec_dir)

        print('Create test archive... ', end='')

        exec(f'np.savez("test", {exec_dir})')

        print('finished')



    def createZip(self):

        if not os.path.exists('labelsDict.npy'):
            self.createDict()
        
        labelDictionary = np.load('labelsDict.npy', allow_pickle=True).tolist()

        trFiles =  self.createTrain((224, 224), 282, labelDictionary)

        tsFiles = self.createTest((224, 224))
