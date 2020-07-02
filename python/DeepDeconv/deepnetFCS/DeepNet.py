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
from keras import metrics
from DeepDeconv.utils.batch_utils import get_batch_from_fits, dynamic_batches, npy_batches
import tensorflow as tf


#Used for loading the model
import json
import h5py
import keras.optimizers as optimizers
from keras.layers import Input
from keras.utils.io_utils import H5Dict


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
        

class ModelCheckpointExtraSave(ModelCheckpoint):
    """ModelCheckpoint wiht extra information."""
  
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto',nepochs=1, period=1,best_epoch=None,best_val=None):
        super(ModelCheckpointExtraSave, self).__init__(filepath, monitor=monitor, verbose=verbose,
              save_best_only=save_best_only, save_weights_only=save_weights_only,
              mode=mode, period=period)
        self.nepochs=nepochs
        if (best_epoch!=None) and (best_val!=None):
            self.best=best_val
            self.best_epoch=best_epoch
            
        
    def on_epoch_end(self,epoch,logs=None):
        """This is essentially the same as ModelCheckpoint, except for the 2 np.savetxt"""
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        self.best_epoch=epoch + 1
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                            np.savetxt(filepath+".best_params",np.asarray([self.best_epoch,self.best,self.nepochs]))
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
                    self.best_epoch=epoch + 1
                    np.savetxt(filepath+".best_params",np.asarray([self.best_epoch,self.best,nmax_epoch,self.nepochs]))


#%%DATA INITIALIZATION
import numpy as np
from AlphaTransform import AlphaShearletTransform as AST
import shape_constraint.cadmos_lib as cl
import os


row,column = np.array([96,96])
U = cl.makeUi(row,column)

# Get shearlet elements
#Step 1 : create a shearlet transform instance
trafo = AST(column, row, [0.5]*3,real=True,parseval=True,verbose=False)
#Step 2 : get shearlets filters
shearlets = trafo.shearlets
#Step 3 : get the adjoints
adjoints = cl.get_adjoint_coeff(trafo)

#Normalize shearlets filter banks
#/!\ The order is important/!\
adjoints = cl.shear_norm(adjoints,shearlets)
shearlets = cl.shear_norm(shearlets,shearlets)

#Compute moments constraint normalization coefficients
#the $\Psi^*_j$ are noted adj_U
adj_U = cl.comp_adj(U,adjoints).reshape(6,27,1,96,96,1)
mu = cl.comp_mu(adj_U)

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
    
    def __init__(self, network_name = 'DNN', img_rows = 96, img_cols = 96, model_file='', verbose=False,shape_constraint=False, gamma=0,shearlet=False):
        self.network_name = network_name
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.U=makeUi(img_rows,img_cols)
        self.shape_constraint=shape_constraint
        self.gamma=gamma
        self.model_file=model_file
        self.shearlet=shearlet
        self.build_model(model_file, verbose)
  
    
    def build_model(self, model_file = '', verbose = False):
        if model_file == '':
            raise ValueError('No model provided')
        else:           
            print('Loading model...')
            print(model_file)
            print('Renaming as...')
            if self.network_name=="DNN":
                new_name=model_file.rsplit(".hdf5")[0]
                self.network_name = new_name
            print(self.network_name)
            if self.shearlet:
                #Load the structure of the model
                custom_objects={'shearlet_loss': self.shearlet_loss,'shearlet_metric':self.shearlet_metric}
                self.model = load_model(model_file, custom_objects=custom_objects,compile=True)
            if not self.shearlet and not self.shape_constraint:
                self.model = load_model(model_file,compile=True)
            if self.shape_constraint:
                #Load the structure of the model
                custom_objects={'shape_loss': self.shape_loss,'shape_metric':self.shape_metric}
                self.model = load_model(model_file, custom_objects=custom_objects,compile=False)
                #The non-connected to output  input placeholder layers are not present. Need to add them and register them
                #START with window
                window_layer=Input(shape=(self.img_rows, self.img_cols,1),name='window')
                self.model.inputs.append(window_layer)
                self.model.input_names.append("window")          
                self.model._feed_inputs.append(window_layer)
                self.model._feed_input_names.append("window")
                self.model._feed_input_shapes.append(K.int_shape(window_layer))
                #Then with norm
                norm_layer=Input(shape=(6, 1,1),name='norm')
                self.model.inputs.append(norm_layer)           
                self.model.input_names.append("norm")
                self.model._feed_inputs.append(norm_layer)
                self.model._feed_input_names.append("norm")
                self.model._feed_input_shapes.append(K.int_shape(norm_layer))

                #Now we need to compile the model
                def convert_custom_objects(obj):
                    """Handles custom object lookup.
                    # Arguments
                        obj: object, dict, or list.
                    # Returns
                        The same structure, where occurrences
                            of a custom object name have been replaced
                            with the custom object.
                    """
                    if isinstance(obj, list):
                        deserialized = []
                        for value in obj:
                            deserialized.append(convert_custom_objects(value))
                        return deserialized
                    if isinstance(obj, dict):
                        deserialized = {}
                        for key, value in obj.items():
                            deserialized[key] = convert_custom_objects(value)
                        return deserialized
                    if obj in custom_objects:
                        return custom_objects[obj]
                    return obj
                #Now we update all optimization parameters (compile=True)
                h5dict=H5Dict(model_file)
                training_config = h5dict.get('training_config')
                if training_config is None:
                    warnings.warn('No training configuration found in save file: '
                              'the model was *not* compiled. '
                              'Compile it manually.')
                else:
                    training_config = json.loads(training_config.decode('utf-8'))
                    optimizer_config = training_config['optimizer_config']
                    optimizer = optimizers.deserialize(optimizer_config,
                            custom_objects=custom_objects)
                    # Recover loss functions and metrics.
                    loss = convert_custom_objects(training_config['loss'])
                    net_metrics = convert_custom_objects(training_config['metrics'])
                    if len(net_metrics)==0:
                        net_metrics=[metrics.mse,self.shape_metric]
                    sample_weight_mode = training_config['sample_weight_mode']
                    loss_weights = training_config['loss_weights']
                    # Compile model.
                    self.model.compile(optimizer=optimizer,
                        loss=loss,
                        weighted_metrics=net_metrics,
                        loss_weights=loss_weights,
                        sample_weight_mode=sample_weight_mode)
                    # Set optimizer weights.
                    if 'optimizer_weights' in h5dict:
                    # Build train function (to get weight updates).
                        self.model._make_train_function()
                        optimizer_weights_group = h5dict['optimizer_weights']
                        optimizer_weight_names = [
                            n.decode('utf8') for n in
                            optimizer_weights_group['weight_names']]
                        optimizer_weight_values = [optimizer_weights_group[n] for n in
                                           optimizer_weight_names]
                    try:
                        self.model.optimizer.set_weights(optimizer_weight_values)
                    except ValueError:
                        warnings.warn('Error in loading the saved optimizer '
                    'state. As a result, your model is '
                    'starting with a freshly initialized '
                    'optimizer.')
                
        if verbose:
            print(self.model.summary())

    def train(self, train_data, model_file = '', epochs=20, batch_size=32, validation_split=0.1, logfile='log.txt'):
        #TO BE UPDATED SOME TIMES FOR SHAPE CONSTRAINT
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
                        noiseless_img_hdu=0, targets_hdu=2, psf_hdu=1,
                        image_dim=96, image_per_row=100,
                        deconv_mode=None, rho_fista=1e-3,
                        risktype="GCV",reg="Dirac",reg_frac=1.0,tol=1e-12,
                        win_filename=None, win_hdu=0, mom_hdu=1,
                        logfile='log.txt',win_validation_filename=None,initial_epoch=0,keep_best_loss=False):
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
                                              risktype=risktype,reg=reg,tol=tol,shape_constraint=self.shape_constraint, 
                                              win_filename=win_validation_filename, win_hdu=0,mom_hdu=1)
        samples_per_epoch = int(len(train_files)*np.ceil(nb_img_per_file/batch_size))
        if keep_best_loss:
            best_params_file=self.model_file.replace(".hdf5",".hdf5.best_params")
            if os.path.isfile(best_params_file): 
                best_epoch,best_val,nepoch=np.loadtxt(best_params_file)
                print("Current best_parameters:",int(best_epoch),best_val)
                model_checkpoint = ModelCheckpointExtraSave(model_file, monitor='val_loss', verbose=1, save_best_only=True,nepochs=epochs,best_epoch=int(best_epoch),best_val=best_val)
            else:
                print("Cannot have access to best parameters for monitor for checkpoint")
                model_checkpoint = ModelCheckpointExtraSave(model_file, monitor='val_loss', verbose=1, save_best_only=True,nepochs=epochs)
        else:
            print("Not using any previous monitored value for checkpoint")
            model_checkpoint = ModelCheckpointExtraSave(model_file, monitor='val_loss', verbose=1, save_best_only=True,nepochs=epochs)

        gen = dynamic_batches(train_files, batch_size=batch_size, nb_img_per_file=nb_img_per_file,
                        noise_std=noise_std, SNR=SNR, noiseless_img_hdu=noiseless_img_hdu, 
                        targets_hdu=targets_hdu, psf_hdu=psf_hdu,
                        image_dim=image_dim, image_per_row=image_per_row,
                        deconv_mode=deconv_mode, rho_fista=rho_fista,
                        risktype=risktype,reg=reg,tol=tol,reg_frac=reg_frac, 
                        shape_constraint = self.shape_constraint, 
                        win_filename=win_filename, win_hdu=0,mom_hdu=1)
        history = self.model.fit_generator(gen, samples_per_epoch=samples_per_epoch, epochs=epochs, 
                                           validation_data=validation_data, verbose=1, 
                                           callbacks=[model_checkpoint, LoggingCallback(filetxt=logfile, log=write_log)],
                                           initial_epoch=initial_epoch)
        return history
    
    def train_generator_npy(self, train_files, validation_file, epochs=20, batch_size=32, 
                                nb_img_per_file=10000, model_file = '', logfile='log.txt'):
        #TO BE UPDATED SOME TIMES FOR SHAPE CONSTRAINT
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
        temp=temp/(self.img_rows*self.img_cols)
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
        temp=0
        for i in range(6):
            temp=temp+self.gamma*mu[:,i,0,0]*(K.square((K.sum(residual*window*self.U[i],axis=(1,2,3)))))/2 
        temp=temp/(self.img_rows*self.img_cols)
        temp=K.expand_dims((K.expand_dims(temp, axis=-1)),axis=-1)
        return temp
    
    def shearlet_loss(self,ytrue,ypred):
        @tf.custom_gradient
        def closs(ypred):
            residual=ypred-ytrue
            temp=0
            temp_grad=0
            loss=K.mean(K.square(ytrue-ypred),axis=-1)
            print('loss',K.int_shape(loss))
            for i in range(6):
                for j in range(27):
                    temp+=mu[i,j]*K.square(K.sum(residual*adj_U[i,j],axis=(1,2,3)))
                    temp_grad+=mu[i,j]*K.sum(residual*adj_U[i,j],axis=(1,2,3))*adj_U[i,j]
            temp=temp*self.gamma/(self.img_rows*self.img_cols)
            temp_grad=temp_grad*self.gamma/(self.img_rows*self.img_cols)
            temp=K.expand_dims((K.expand_dims(temp, axis=-1)),axis=-1)
            temp_grad=K.permute_dimensions(temp_grad,(3,1,2,0))
            print('temp',K.int_shape(temp))
            loss+=temp  
            def grad(dy):
                return (2*(ypred-ytrue)+temp_grad)*K.expand_dims(dy,axis=-1)
            return loss,grad
        loss=closs(ypred)
        print(type(loss))
        return closs(ypred)
    
    def shearlet_metric(self, ytrue,ypred):
        residual=ypred-ytrue
        temp=0
        for i in range(6):
            for j in range(27):
                temp+=mu[i,j]*K.square(K.sum(residual*adj_U[i,j],axis=(1,2,3)))
        temp=temp*self.gamma/(self.img_rows*self.img_cols)
        temp=K.expand_dims((K.expand_dims(temp, axis=-1)),axis=-1)    
        return temp
    
    def custom_mse_3(self,y_true,y_pred):
        print(K.int_shape(y_true),K.int_shape(y_pred))
        @tf.custom_gradient
        def closs(y_pred):
            loss=K.square(y_true-y_pred)
            def grad(dy):
                print(K.int_shape(dy))
                return 2*dy*(y_pred-y_true)
            print(K.int_shape(loss))
            return loss,grad
        return closs(y_pred)
