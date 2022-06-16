from keras.applications.vgg16 import preprocess_input
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.math_ops import multiply as ops_mult
from tensorflow.python.ops.array_ops import stop_gradient as stop_grad
import time
from tqdm import tqdm

# ------------------------- #

class TqdmExtraFormat(tqdm):
    
    @property
    def format_dict(self):
        d = super(TqdmExtraFormat, self).format_dict
        total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
        d.update(total_time=self.format_interval(total_time) + " est.")
        return d

# ------------------------- #

class InfluenceCalculation:

    def __init__(
        
            self, 
            model=None, 
            loss=lambda l, p: tf.norm(l - p), 
            damping=1e-4, 
            n_classes=1000, 
            preprocess_input=preprocess_input
        
        ):

        if model is None:
            raise ValueError('model cannot be NoneType') # raise error if no model
            
        self.model = model
        self.weights = model.trainable_variables
        self.len_weights = len(self.weights)
        self.loss = loss
        self.damping = damping
        self.n_classes = n_classes
        self.preprocess_input = preprocess_input
        self.test_grad = None
    
    # ------------------------- #
    
    # elemwise product with gradient stopping
    def elementwise_products(self, grad, eigv):
        return [ops_mult(g, stop_grad(v)) for g, v in zip(grad, eigv) if g is not None]
    
    # ------------------------- #
    
    # get gradient on prediction with gradient vanishing
    def gradient(self, data, label, weights, training=True):
        with tf.GradientTape(persistent=True) as t:
            predict = self.model(data) # take predict from model related to instance
            loss = self.loss(label, predict) # calculate basic loss 
        return t.gradient(loss, weights) 
    
    # ------------------------- #

    # filtering None
    def replace_nones(self, grads):
        return [g if g is not None else tf.zeros_like(p) for p, g in zip(self.weights, grads)]
    
    # ------------------------- #

    # calculate gradients separately for each trainable variable in NN
    def separate_gradients(self, tape, ep):
        sep_grds = []
        for i in range(self.len_weights):
            grd = tape.gradient(ep[i], self.weights[i])
            ep[i] = None # for used RAM reducing
            sep_grds.append(grd[0])
        del ep
        return sep_grds
    
    # ------------------------- #

    # hvp with parallel allowing
    def hessian_vector_products(self, data, labels, eigv, parallel):
        grads_no_nones = []
        if parallel: # more memory required, works 1.5-2 times faster
            with tf.GradientTape(persistent=True) as t0:
                with tf.GradientTape(persistent=True) as t1:
                    predicts = self.model(data)
                    loss = tf.reduce_mean(self.loss(labels, predicts))

                grad1 = t1.gradient(loss, self.weights) # calculate 1st order gradients
                t0.watch(grad1)
                elemwise_products = self.elementwise_products(grad1, eigv)
                
            #separate_grads = self.separate_gradients(t0, elemwise_products) # calculate 2nd order gradients separately
            separate_grads = t0.gradient(elemwise_products, self.weights) # calculate 2nd order gradients

            del elemwise_products # clear memory
            grads_no_nones = self.replace_nones(separate_grads) # replace None elems with zeros with the same shape to avoid errors
            del separate_grads # clear memory
            return grads_no_nones
        
        else: # less memory needed
            for weight, eigval in zip(self.weights, eigv): # for each trainable parameter
                with tf.GradientTape(persistent=True) as t0:
                    with tf.GradientTape(persistent=True) as t1:
                        predicts = self.model(data)
                        loss = tf.reduce_mean(self.loss(labels, predicts))

                    grad1 = t1.gradient(loss, weight) # calculate 1st order gradient
                    t0.watch(grad1)
                    del predicts
                    elemwise_products = ops_mult(grad1, stop_grad(eigval)) if grad1 is not None else None

                separate_grads = t0.gradient(elemwise_products, weight) # calculate 2nd order gradient
                separate_grads = tf.zeros_like(weight) if separate_grads is None else separate_grads # replace None elems with zeros
                del elemwise_products
                grads_no_nones.append(separate_grads)
                del separate_grads
            return grads_no_nones
    
    # ------------------------- #

    # one iteration of ihvp calculating
    def ihvp_iteration(self, data, labels, cur_v, test_gradient, batch_size, batch_range, scale, parallel):
        rand_idx = np.random.randint(batch_range, size=[batch_size]) # get a random core for stochastic estimation
        iter_grads = self.hessian_vector_products(data[rand_idx], labels[rand_idx], cur_v, parallel=parallel) # iterate on the rand core
        new_ihvp = [g + self.damping*v - hv / scale for g, hv, v in zip(test_gradient, iter_grads, cur_v)] # calculate new ihvp tensors

        return new_ihvp

    # ------------------------- #

    def to_categorical(self, label):
        output = []
        for l in label:
            enc = np.zeros((self.n_classes))
            enc[int(l)] = 1
            output.append(enc)
        return np.array(output, dtype=np.int32)
    
    # ------------------------- #

    # main function of ihvp calculating
    def inverse_hessian_vector_products(self, train, n_batches, test_data, test_label,
                                       num_iter=101, stochast_batch_size=8,
                                       scale=1e0, parallel=True):
        #start_time = time.time() # count all time
        
        print('Calculating gradients for test image...', end=' ')
        
        self.test_grad = self.gradient(test_data, test_label, self.weights) # calculate gradients for particular

        print('finished')
        
        time.sleep(0.5) # sleep to avoid early print 
        current_v = self.test_grad # init current v for hv calculation as test grad
        current_batch = np.random.randint(n_batches)
        data = self.preprocess_input(train[f'train_x_{current_batch}'])
        labels = self.to_categorical(train[f'train_y_{current_batch}'])
        current_ihvp = None
        stochast_batch_range = len(data)
        
        # TqdmExtraFormat as wrapper for tqdm progress bar to display progress of ihvp calculation/estimation

        for j in TqdmExtraFormat(iterable=range(num_iter), ncols=100, ascii=' ▮', desc='IHVP calculation',
                        bar_format="{desc} | {total_time} | {percentage:.0f}% |{bar}{r_bar}"):

            current_ihvp = self.ihvp_iteration(data, labels, current_v, self.test_grad,
                                         stochast_batch_size, stochast_batch_range,
                                         scale,  parallel=parallel)
            current_v = current_ihvp
        
        current_ihvp = [e / scale for e in current_ihvp]
        #total_time = time.time() - start_time # calculate full time

        del data, labels
        
        return current_ihvp
    
    # ------------------------- #

    # calculate ihvp for each layer (wrapper)
    def calculate(self, train, n_batches, test_data, test_label, num_iter=101, 
                  stochast_batch_size=8, scale=1e3, parallel=True):
        
        ihvp = self.inverse_hessian_vector_products(train, n_batches, test_data, 
                                                    test_label, num_iter,
                                                    stochast_batch_size, 
                                                    scale=scale, parallel=parallel)
        return ihvp, self.test_grad

    # ------------------------- #

    #helper makes elements in tensor (0 axis) flatten
    def flatten_tensor_elements(self, tensor):
        return [tf.reshape(a, (-1,)) for a in tensor]

    # ------------------------- #

    #helper makes full flat tensor
    def flatten_tensor(self, tensor):
        flat_elems = self.flatten_tensor_elements(tensor)
        return tf.concat(flat_elems, 0)