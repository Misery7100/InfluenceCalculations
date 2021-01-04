from classes.influence_calc import InfluenceCalc, TqdmExtraFormat
from classes.label_subset import LabelSubset
from classes.influence_plot import InfluencePlot
import tensorflow as tf
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import os
import time


class Influence:
    
    def __init__(self, npz, model=VGG16(), x_preproc=preprocess_input, n_classes=1000):
        self.model = model
        self.x_preproc = x_preproc
        self.n_classes = n_classes
        self.npz = npz
        print('\nExtracting data, creating instance\n')
        self.extract_data()
        self.init_calc_instance()
        
    def to_categorical(self, label):
        output = []
        for l in label:
            enc = np.zeros((self.n_classes))
            enc[l] = 1
            output.append(enc)
        return np.array(output, dtype=np.int32)
    
    def extract_data(self):
        data_to_extract = np.load(self.npz)
        for k in data_to_extract.files[:4]:
            exec(f'self.{k} = data_to_extract["{k}"]')
        
        self.unique_test_labels = np.unique(self.test_y)
        
        for label in self.unique_test_labels:
            if not os.path.exists(f'{label}.npz'):
                ls = LabelSubset(label=label)
                ls.subsetByLabels(data_to_extract)

        self.randcore_x = self.x_preproc(self.randcore_x)
        
        self.randcore_y = self.to_categorical(self.randcore_y)
        

                
    def init_calc_instance(self, damping=1e-2):
        self.calc_instance = InfluenceCalc(self.model, damping=damping)
    
    def calculate_single_label(self, label, num_iter=30, batch_size=8, save_format='pdf'):
        custom_data = np.load(f'{label}.npz')
        ls = LabelSubset(label=label)
        test_x, test_y = ls.extractLabels(self.test_x, self.test_y), ls.extractLabels(self.test_y, self.test_y)

        test_x = self.x_preproc(test_x)
        test_y = self.to_categorical(test_y)
        
        influence_data, influence_labels = {}, {}
        for i in range(len(custom_data.files[2::2])):
            x, y = custom_data.files[2::2][i], custom_data.files[3::2][i]
            influence_data[x] = np.concatenate([custom_data['orig_x'], custom_data[x]])
            influence_labels[y] = np.concatenate([custom_data['orig_y'], custom_data[y]])
        
        print('Preprocessing data... ', end='')

        for i, j in zip(influence_data.keys(), influence_labels.keys()):
            influence_data[i] = self.x_preproc(influence_data[i])
            influence_labels[j] = self.to_categorical(influence_labels[j])

        print('finished\n\n')
        
        for key_x, key_y in zip(influence_data.keys(), influence_labels.keys()):
            print(f'Calculating influence for {key_x[:-2]} data, label: {label}\n')
            for i in range(test_y.shape[0]):
                print(f'Test image #{i + 1} in process')
                test_data = np.expand_dims(test_x[i], axis=0)
                test_label = np.expand_dims(test_y[i], axis=0)

                plot = InfluencePlot(self.calc_instance, train_data=self.randcore_x, train_labels=self.randcore_y, 
                                 test_data=test_data, test_label=test_label, 
                                 influence_data=influence_data[key_x], influence_labels=influence_labels[key_y])

                plot.calculate_ihvp(num_iter=num_iter, scale=1e4, sb_size=batch_size)
                plot.get_influences(name=f'{key_x[:-2]}_{i+1}', format=save_format, save_to=f'{label}')
                del test_data, test_label, plot

        del custom_data, test_x, test_y, influence_data, influence_labels, ls
    
    def calculate_all_influences(self, num_iter=30, batch_size=8, save_format='pdf'):
        
        print('\nStart calculating all influences', end='\n\n')
        time.sleep(0.5)
        for label in self.unique_test_labels:
            self.calculate_single_label(label, num_iter=num_iter, batch_size=batch_size, save_format=save_format)