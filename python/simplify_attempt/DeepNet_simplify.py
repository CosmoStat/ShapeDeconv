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

from simplify_attempt.batch_utils_simplify import get_batch_from_fits, dynamic_batches
import shape_constraint.cadmos_lib as clw
from keras import metrics

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


'''def custom_loss(y_true, y_pred):
    weights = y_true[:,:,:,1]
    y_true = y_true[:,:,:,0]
    return K.mean(K.tf.multiply(weights, K.square(y_pred - y_true)), axis=-1)'''
    

def comp_loss(residual,gamma,mu,gauss_win,U):
    return norm(residual)**2/2.+gamma*(np.array(
            [m*((residual*gauss_win*u).sum())**2
            for m,u in zip(mu,U)])/2.).sum()

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

def makeU1(n,m):
    """Create a n x m numpy array with (i)_{i,j} entries where i is the ith
    line and j is the jth column
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U1 = np.tile(np.arange(n),(m,1)).T
    return U1


def makeU3(n,m):
    """Create a n x m numpy array with (1)_{i,j} entries where i is the ith
    line and j is the jth column
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U3 = np.ones((n,m))
    U3=add_extra_dimension(U3)
    return U3

def makeU6(n,m):
    """Create a n x m numpy array with (i*j)_{i,j} entries where i is the ith
    line and j is the jth column
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U6 = np.outer(np.arange(n),np.arange(m))
    U6=add_extra_dimension(U6)
    return U6

def add_extra_dimension(U1):
    lns=tuple(list(np.shape(U1))+[1])
    return np.reshape(U1,lns)
    

def makeUi(n,m):
    """Create a 6 x n x m numpy array containing U1, U2, U3, U4, U5 and U6
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: 6 x n x m numpy array"""
    U1 = makeU1(n,m)
    Ul = U1**2
    Uc = Ul.T
    U1T=U1.T
    U1=add_extra_dimension(U1)
    U1T=add_extra_dimension(U1T)
    Uc=add_extra_dimension(Uc)
    Ul=add_extra_dimension(Ul)
    return np.array([U1,U1T,makeU3(n,m),Ul+Uc,Ul-Uc,makeU6(n,m)])


class DeepNet(object):
    
    def __init__(self, network_name = 'DNN', img_rows = 96, img_cols = 96, model_file='', verbose=False,shape_constraint=False, gamma=None,initial_epoch=0):
        self.network_name = network_name
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.build_model(model_file, verbose)
        self.U=makeUi(img_rows,img_cols)
        self.shape_constraint=shape_constraint
        self.gamma=gamma
        self.initial_epoch=initial_epoch
    
    def build_model(self, model_file = '', verbose = False):
        if model_file == '':
            raise ValueError('No model provided')
        else:
            print('Loading model in DeepNet...')
            print(model_file)
            self.model = load_model(model_file, custom_objects={"shape_loss":self.shape_loss},compile=False)
        if verbose:
            print(self.model.summary())


    def train_generator(self, train_files, validation_file, epochs=20, batch_size=32, model_file = '',
                        nb_img_per_file=10000, validation_set_size=10000,
                        noise_std=None, SNR=None,
                        noiseless_img_hdu=0, targets_hdu=0, psf_hdu=0,
                        image_dim=96, image_per_row=100,
                        deconv_mode=None, rho_fista=1e-3,
                        risktype="GCV",reg="Dirac",reg_frac=1.0,tol=1e-12, 
                        win_filename=None, win_hdu=0, mom_hdu=1,
                        logfile='log.txt',win_validation_filename=None,initial_epoch=0):
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
                                              deconv_mode=deconv_mode, rho_fista=rho_fista,
                                              risktype=risktype,reg=reg,tol=tol,
                                              shape_constraint=self.shape_constraint, 
                                              win_filename=win_validation_filename, win_hdu=0,mom_hdu=1)
        samples_per_epoch = int(len(train_files)*np.ceil(nb_img_per_file/batch_size))
        model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True)
        gen = dynamic_batches(train_files, batch_size=batch_size, nb_img_per_file=nb_img_per_file,
                        noise_std=noise_std, SNR=SNR,
                        noiseless_img_hdu=noiseless_img_hdu, targets_hdu=targets_hdu, psf_hdu=psf_hdu,
                        image_dim=image_dim, image_per_row=image_per_row,
                        deconv_mode=deconv_mode, rho_fista=rho_fista,
                        risktype=risktype,reg=reg,tol=tol,reg_frac=reg_frac,
                        shape_constraint = self.shape_constraint, 
                        win_filename=win_filename, win_hdu=0,mom_hdu=1)
        history = self.model.fit_generator(gen, samples_per_epoch=samples_per_epoch, epochs=epochs, 
                                           validation_data=validation_data, verbose=1, 
                                           callbacks=[model_checkpoint, LoggingCallback(filetxt=logfile,
                                           log=write_log)],initial_epoch=initial_epoch)
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
    
    def shape_loss(self,y_true,y_pred):
        #window = y_true[1]
        #mu=y_true[2]
        #print(K.int_shape(y_true),K.int_shape(y_pred))
        residual=y_true-y_pred
        M=K.mean(K.square(y_true-y_pred), axis=-1)
        window=self.model.input[1]
        mu=self.model.input[2]
        print(M[0],window,mu,residual)
        #print(K.eval(mu[0,0,:,:]))
        #M1=K.eval(M[0])
        #print("MSE=",K.int_shape(M))
        #print(self.model.input[1])
        #print(self.model.input[2])
        #print('WIN=',K.int_shape(window),'\n','RES=',K.int_shape(residual),'\n','U=',np.shape((self.U)[0]),'\n','MU=',K.int_shape(mu[:,0,:,:]))
        #print(K.sum(y_true * y_pred, axis=-1))
        #for i in range(6):
            #M=M+self.gamma*mu[:,i,:,:]*K.square(K.sum(residual*window*self.U[i],axis=-1))/2.0
        temp=0
        for i in range(6):
            temp+=self.gamma*mu[:,i,0,0]*(K.square((K.sum((residual)*window*self.U[i],axis=(1,2,3)))))/2
        #print("MSE+SHAPE",K.int_shape(K.expand_dims(temp, axis=-1)))
        temp=K.expand_dims((K.expand_dims(temp, axis=-1)),axis=-1)
        #print(M1,'\n',"MSE+SHAPE",K.int_shape(M1))
        return M+temp
            
#            np.array(
#                [m*((residual*window*u).sum())**2
#                for m,u in zip(mu,self.U)])/2.).sum()

    def shape_metric(self,y_true,y_pred):
        temp=0
        residual=y_true-y_pred
        window=self.model.input[1]
        mu=self.model.input[2]
        shape_loss=0
        for i in range(6):
            shape_loss=shape_loss+self.gamma*mu[:,i,0,0]*(K.square((K.sum(residual*window*self.U[i],axis=(1,2,3)))))/2 
        return K.mean(shape_loss)
