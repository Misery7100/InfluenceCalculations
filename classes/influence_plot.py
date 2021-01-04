from classes.influence_calc import InfluenceCalc, TqdmExtraFormat
from classes.label_subset import LabelSubset
import tensorflow as tf
import os
import time

import matplotlib.pyplot as plt
import numpy as np

class InfluencePlot:
    
    def __init__(self, infcalc, train_data, train_labels, test_data, test_label, influence_data, influence_labels):
        self.instance = infcalc
        self.train_images = {'data': train_data, 'labels': train_labels}
        self.test_image = {'data': test_data, 'labels': test_label}
        self.influence_images = {'data': influence_data, 'labels': influence_labels}
        self.ihvp = None
        self.test_grads = None
    
    def calculate_ihvp(self, num_iter=21, scale=1e3, sb_size=8, parallel=True):
        ihvp, _, test_grads = self.instance.calculate(data=self.train_images['data'],
                                                   labels=self.train_images['labels'],
                                                   test_data=self.test_image['data'],
                                                   test_label=self.test_image['labels'],
                                                   num_iter=num_iter, scale=scale,
                                                   stochast_batch_size=sb_size,
                                                   parallel=parallel)
        
        self.ihvp = self.instance.flatten_tensor_elements(ihvp)
        self.test_grads = self.instance.flatten_tensor_elements(test_grads)
    
    def calculate_eigenvalues(self):
        named_w = [(w.name.split('/')[0], ihvp, tg) for w, ihvp, tg in zip(self.instance.weights, self.ihvp, self.test_grads)]
        ws = dict()
        for w in named_w:
            key = ws.get(w[0], None)
            if key is None: ws[w[0]] = (w[1], w[2])
            else:
                ihvp, tg = ws[w[0]]
                ihvp = tf.concat([ihvp, w[1]], 0)
                tg = tf.concat([tg, w[2]], 0)
                ws[w[0]] = (ihvp, tg)
                
        eigv = dict()
        for key in ws.keys():
            ihvp, tg = ws[key]
            eigv[key] = np.linalg.norm(ihvp) / np.linalg.norm(tg)
            
        return eigv
    
    def calculate_influences(self):
        influences = dict()
        data, labels = self.influence_images['data'], self.influence_images['labels']
        time.sleep(0.5)
        for i, image in enumerate(zip(TqdmExtraFormat(iterable=data, ncols=100, ascii=' ▮', desc='Inf. calculation',
                        bar_format="{desc} | {total_time} | {percentage:.0f}% |{bar}{r_bar}"), labels)):
            
            xt, yt = image
            xt = np.expand_dims(xt, axis=0)
            yt = yt.reshape(1, -1)
            
            train_grad = self.instance.gradient(xt, yt, self.instance.weights)
            train_grad = self.instance.flatten_tensor_elements(train_grad)
            named_w = [(w.name.split('/')[0], ihvp, tg) for w, ihvp, tg in zip(self.instance.weights, self.ihvp, train_grad)]
            
            ws = dict()
            for w in named_w:
                key = ws.get(w[0], None)
                if key is None: ws[w[0]] = (w[1], w[2])
                else:
                    ihvp, tg = ws[w[0]]
                    ihvp = tf.concat([ihvp, w[1]], 0)
                    tg = tf.concat([tg, w[2]], 0)
                    ws[w[0]] = (ihvp, tg)

            infs = dict()
            total = 0
            for key in ws.keys():
                ihvp, tg = ws[key]
                product = np.dot(ihvp, tg)
                infs[key] = product
                total += product
            infs['total'] = total
            influences[i] = infs
            
        return influences
    
    # Restore from BGR image (vgg preproc) to RGB
    def restore_image(self, img):
        mean = [103.939, 116.779, 123.68]
        img = img.copy()
        # Zero-center by mean pixel
        img[..., 0] += mean[0]
        img[..., 1] += mean[1]
        img[..., 2] += mean[2]

        img = img[..., ::-1]

        return img.astype(np.int16)
    
    def get_influences(self, top=5, name='sample_name', format='png', save_to='sample_folder'):
        influences = self.calculate_influences()
        sort_infl = dict(sorted(influences.items(), key=lambda d: d[1]['total']))
        
        # Harmful
        figh, axh = plt.subplots(1, top + 1, figsize=(20, 6))
        
        figh.suptitle(f'Top {top} harmful images', fontsize=14)
        
        for a in axh: a.axis('off')
        axh[-1].imshow(self.restore_image(self.test_image['data'][0]))
        axh[-1].set_title('Test image')
        for j, i in enumerate(list(sort_infl.keys())[:top]):
            axh[j].imshow(self.restore_image(self.influence_images['data'][i]))
            axh[j].set_title(f'Influence: {influences[i]["total"]:.5f}', pad=10)
        
        # Useful
        figu, axu = plt.subplots(1, top + 1, figsize=(20, 6))
        
        figu.suptitle(f'Top {top} useful images', fontsize=14)
        
        for a in axu: a.axis('off')
        axu[-1].imshow(self.restore_image(self.test_image['data'][0]))
        axu[-1].set_title('Test image')
        for j, i in enumerate(list(sort_infl.keys())[-top:][::-1]):
            axu[j].imshow(self.restore_image(self.influence_images['data'][i]))
            axu[j].set_title(f'Influence: {influences[i]["total"]:.5f}', pad=10)
        
        # safely create a nested directory
        if not os.path.exists(save_to):
            os.makedirs(save_to)
            
        figh.savefig(f'{save_to}/harmful_{name}.{format}')
        figu.savefig(f'{save_to}/useful_{name}.{format}')
        np.save(f'{save_to}/{name}.npy', influences)
        
        print(f'Top {top} harmful\t\t->', f'{save_to}/harmful_{name}.{format}', sep='\t')
        print(f'Top {top} useful\t\t->', f'{save_to}/useful_{name}.{format}', sep='\t')
        print(f'All influences dict\t->', f'{save_to}/{name}.npy', sep='\t', end='\n\n')
        
        return influences