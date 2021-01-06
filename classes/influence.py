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
import shutil


class Influence:
    
    def __init__(self, train_npz, test_npz, model, x_preproc=preprocess_input, n_classes=1000):
        self.model = model
        self.x_preproc = x_preproc
        self.n_classes = n_classes
        self.train_npz = train_npz
        self.test_npz = test_npz
        self.extract_data()
        self.init_calc_instance()
        
    def to_categorical(self, label):
        output = []
        for l in label:
            enc = np.zeros(self.n_classes)
            enc[l] = 1
            output.append(enc)
        return np.array(output, dtype=np.int32)
    
    def extract_data(self):
        self.train_npz = np.load(self.train_npz)
        self.test_npz = np.load(self.test_npz)
        self.n_batches = int(len(self.train_npz.files) / 2)
    
    def calculate_folder(self, folder, num_iter, batch_size, save_format, scale=1e3):
        print(f'Calculating influences for folder: {folder}')
        
        folder_data = self.test_npz[f'test_{folder}_x']
        folder_labs = self.test_npz[f'test_{folder}_y']
        folder_unique_labels = np.unique(folder_labs)

        output_base = 'calc_output'

        if not os.path.exists(output_base):
            os.mkdir(output_base)
        
        output_name = f'{output_base}/{folder}'

        if not os.path.exists(output_name):
            os.mkdir(output_name)
                     
        for label in folder_unique_labels:
            if not os.path.exists(f'{output_name}/{label}.npz'):
                ls = LabelSubset(label=label, folder=output_name)
                ls.subsetByLabels(folder_data, folder_labs)
        
        del folder_data, folder_labs
                     
        for label in folder_unique_labels:
            custom_data = np.load(f'{output_name}/{label}.npz')
            test_x, test_y = custom_data['x'], custom_data['y'].astype(np.int32)
            test_x = self.x_preproc(test_x)
            labels = test_y.copy()
            test_y = self.to_categorical(test_y)
                     
            print(f'Calculating influences for class: {label}\n')
                     
            for i in range(test_y.shape[0]):
                print(f'Test image #{i + 1} in process')
                test_data = np.expand_dims(test_x[i], axis=0)
                test_label = np.expand_dims(test_y[i], axis=0)
                label_raw = labels[i]

                plot = InfluencePlot(self.calc_instance, 
                                     train=self.train_npz,
                                     test_data=test_data, 
                                     test_label=test_label,
                                     label=label_raw,
                                     n_batches=self.n_batches)

                plot.calculate_ihvp(num_iter=num_iter, scale=scale, sb_size=batch_size)
                plot.get_influences(name=f'{i+1}', format=save_format, save_to=f'{output_name}/{label}')
                del test_data, test_label, plot
            
            
            path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, f'{output_name}/{label}.npz'))
            os.remove(path)
                
    def init_calc_instance(self, damping=1e-2):
        self.calc_instance = InfluenceCalc(self.model, damping=damping)
    
    def calculate_all_influences(self, num_iter=30, batch_size=8, save_format='pdf', scale=1e3):
        
        print('\n\nStart calculating all influences', end='\n\n')
        time.sleep(0.5)
        all_folders = [i.split('_')[1] for i in self.test_npz.files[::2]]
        for folder in all_folders:
            self.calculate_folder(folder=folder, num_iter=num_iter, 
                                    batch_size=batch_size, save_format=save_format,
                                    scale=scale)