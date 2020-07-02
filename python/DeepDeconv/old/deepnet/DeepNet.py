#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 13:53:31 2018

@author: alechat
"""
import os, sys
if os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    
import numpy as np
import datetime as dt
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects

from DeepDeconv.utils.batch_utils import get_batch_from_fits, dynamic_batches, npy_batches


# Write to a file
def write_log(s, filetxt):
    with open(filetxt, 'a') as f:
        f.write(s)
        f.write("\n")

class LoggingCallback(Callback):
    """Callback that logs message at end of epoch."""

    def __init__(self, filetxt='log.txt', log=write_log):
        Callback.__init__(self)
        self.log = log
        self.filetxt = filetxt

    def on_epoch_end(self, epoch, logs={}):
        msg = dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S - ') + str("Epoch: %i, "%(epoch+1)) + str(", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.log(msg, self.filetxt)
        print(msg)


def custom_loss(y_true, y_pred):
    weights = y_true[:,:,:,1]
    y_true = y_true[:,:,:,0]
    return K.mean(K.tf.multiply(weights, K.square(y_pred - y_true)), axis=-1)

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': swish})

def get_model_memory_usage(batch_size, model):
    '''Compute memory usage for the model and one batch of data'''
    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


class DeepNet(object):
    
    def __init__(self, network_name = 'DNN', img_rows = 96, img_cols = 96, model_file='', verbose=False):
        self.network_name = network_name
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.build_model(model_file, verbose)
    
    def build_model(self, model_file = '', verbose = False):
        if model_file == '':
            raise ValueError('No model provided')
        else:
            print('Loadind model...')
            print(model_file)
            self.model = load_model(model_file, custom_objects={'custom_loss': custom_loss})
        if verbose:
            print(self.model.summary())

    def train(self, train_data, model_file = '', epochs=20, batch_size=32, validation_split=0.1, logfile='log.txt'):
        if self.model is None:
            raise Exception("No model found, please use build_model()")
        if model_file == '':
            model_file = self.network_name + '.hdf5'
            print('Model will be saved at %s/%s'%(os.getcwd(), model_file))
        model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss',verbose=1, save_best_only=True)
        print('Fitting model...')
        self.model.fit(train_data[0], train_data[1], batch_size=batch_size, epochs=epochs, verbose=1, 
                       validation_split=validation_split, shuffle=True, 
                       callbacks=[model_checkpoint, LoggingCallback(filetxt=logfile, log=write_log)])

    def train_generator(self, train_files, validation_file, epochs=20, batch_size=32, model_file = '',
                        nb_img_per_file=10000, validation_set_size=10000,
                        noise_std=None, SNR=None,
                        noiseless_img_hdu=0, targets_hdu=0, psf_hdu=0,
                        image_dim=96, image_per_row=100,
                        deconv_mode=None, rho_fista=1e-3,
                        logfile='log.txt'):
        if self.model is None:
            raise Exception("No model found, please use build_model()")
        if model_file == '':
            model_file = self.network_name + '.hdf5'
            print('Model will be saved at %s/%s'%(os.getcwd(), model_file))
        print('Memory usage for the model + one batch (GB): %f'%(get_model_memory_usage(batch_size, self.model)))
        with open(logfile, 'a') as f:
            f.write(self.network_name)
            f.write("\n")
        validation_data = get_batch_from_fits(validation_file,
                                              idx_list=np.arange(validation_set_size), 
                                              noise_std=noise_std, SNR=SNR,
                                              noiseless_img_hdu=noiseless_img_hdu, 
                                              targets_hdu=targets_hdu, psf_hdu=psf_hdu,
                                              image_dim=image_dim, image_per_row=image_per_row,
                                              deconv_mode=deconv_mode, rho_fista=rho_fista)
        samples_per_epoch = int(len(train_files)*np.ceil(nb_img_per_file/batch_size))
        model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True)
        gen = dynamic_batches(train_files, batch_size=batch_size, nb_img_per_file=nb_img_per_file,
                        noise_std=noise_std, SNR=SNR,
                        noiseless_img_hdu=noiseless_img_hdu, targets_hdu=targets_hdu, psf_hdu=psf_hdu,
                        image_dim=image_dim, image_per_row=image_per_row,
                        deconv_mode=deconv_mode, rho_fista=rho_fista)
        history = self.model.fit_generator(gen, samples_per_epoch=samples_per_epoch, epochs=epochs, 
                                           validation_data=validation_data, verbose=1, 
                                           callbacks=[model_checkpoint, LoggingCallback(filetxt=logfile, log=write_log)])
        return history
    
    def train_generator_npy(self, train_files, validation_file, epochs=20, batch_size=32, nb_img_per_file=10000, model_file = '', logfile='log.txt'):
        if self.model is None:
            raise Exception("No model found, please use build_model()")
        if model_file == '':
            model_file = self.network_name + '.hdf5'
            print('Model will be saved at %s/%s'%(os.getcwd(), model_file))
        print('Memory usage for the model + one batch (GB): %f'%(get_model_memory_usage(batch_size, self.model)))
        with open(logfile, 'a') as f:
            f.write(self.network_name)
            f.write("\n")
        validation_data = np.load(validation_file)
        samples_per_epoch = int(len(train_files)*np.ceil(nb_img_per_file/batch_size))
        model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True)
        gen = npy_batches(train_files, batch_size=batch_size, nb_img_per_file=nb_img_per_file)
        history = self.model.fit_generator(gen, samples_per_epoch=samples_per_epoch, epochs=epochs, validation_data=validation_data, verbose=1, callbacks=[model_checkpoint, LoggingCallback(write_log)])
        return history

    def predict(self, test_data, verbose=1):
        if self.model is None:
            raise Exception("No model found, please use build_model()")
        output_test = self.model.predict(test_data, batch_size=1, verbose=verbose)
        return output_test

    def get_layer_output(self, test_data, layer_idx):
        if self.model is None:
            raise Exception("No model found, please use build_model()")
        get_output = K.function([self.model.layers[0].input], [self.model.layers[layer_idx].output])
        return get_output([test_data])[0]
