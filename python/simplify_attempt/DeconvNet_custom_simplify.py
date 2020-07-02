#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:53:50 2018

@author: alechat/fsureau
"""
import os, sys
if os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Concatenate, Conv2D, UpSampling2D, SeparableConv2D, BatchNormalization, AveragePooling2D, Activation, Add
from keras.optimizers import Adam
from keras import metrics 

from  simplify_attempt.DeepNet_simplify import DeepNet,makeUi


def DenseBlock(n_layers, n_kernels, input_layer, activation_function='swish',
                                            axis_concat=3, concat_input=True,atrou=False,dilation_rate=1):
    #concat_input: implies that we have a skip concatenation between input and output of block
    connect_input = input_layer
    for n in range(n_layers):
        bn = BatchNormalization()(connect_input)
        act = Activation(activation_function)(bn)
        if atrou:
            conv = SeparableConv2D(n_kernels, 3, padding='same', use_bias=False,
                               dilation_rate=dilation_rate,kernel_initializer='he_uniform')(act)        
        else:
            conv = SeparableConv2D(n_kernels, 3, padding='same', use_bias=False,
                                        kernel_initializer='he_uniform')(act)
        if n == 0:
            concat = conv
        else:
            concat = Concatenate(axis = axis_concat)([concat, conv])
        connect_input = concat
    if concat_input:
        return Concatenate(axis = axis_concat)([input_layer, concat])
    return concat


class DeconvNet(DeepNet):

    def __init__(self, network_name = 'DeconvNet', img_rows = 96, img_cols = 96, model_file='',
                                    verbose=False, nb_scales=4, growth_rate=12, 
                                    nb_layers_per_block=[4,5,6,7],
                                    activation_function='swish',resNet=False,atrou=False,gamma=None,
                                    shape_constraint=False):
        if len(nb_layers_per_block) != nb_scales:
            raise ValueError('nb_layers_per_block is a list and must contain nb_scales values')
        self.nb_scales = nb_scales
        self.growth_rate = growth_rate
        self.nb_layers_per_block = nb_layers_per_block
        self.activation_function = activation_function
        self.atrou=atrou
        self.resNet=resNet
        self.gamma=gamma
        self.shape_constraint=shape_constraint
        self.U=makeUi(img_rows,img_cols)
        super(DeconvNet, self).__init__(network_name=network_name, img_rows=img_rows, img_cols=img_cols, model_file=model_file, verbose=verbose,gamma=gamma,shape_constraint=shape_constraint) 



    def build_model(self, model_file = '', verbose = False):
        if model_file == '':
            inputs = Input((self.img_rows, self.img_cols, 1))
            window = Input((self.img_rows, self.img_cols,1),name='window')
            norm = Input((6,1,1),name='norm')

            #INPUT CONV
            x = Conv2D(32, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(inputs)

            #CONTRACTING PATH
            skip_connect = []
            if self.resNet:
                skip_connect.append(inputs)#0: skip connect
                offscale=1
            else:
                offscale=0
            for b in range(self.nb_scales-1):
                if self.atrou:         
                    block = DenseBlock(n_layers=self.nb_layers_per_block[b],
                                   n_kernels=self.growth_rate,
                                   input_layer=x,
                                   activation_function=self.activation_function,
                                   concat_input=True,atrou=True,dilation_rate=2**(b))
                else:
                    block = DenseBlock(n_layers=self.nb_layers_per_block[b],
                                   n_kernels=self.growth_rate,
                                   input_layer=x,
                                   activation_function=self.activation_function,
                                   concat_input=True)
                skip_connect.append(block)
                bn = BatchNormalization()(block)
                act = Activation(self.activation_function)(bn)
                if self.atrou:
                    x = Conv2D(32+np.sum(self.nb_layers_per_block[:b+1])*self.growth_rate, 1, padding='same', use_bias=False, kernel_initializer='he_uniform')(act)
                else:
                    conv_transi = Conv2D(32+np.sum(self.nb_layers_per_block[:b+1])*self.growth_rate, 1, padding='same', use_bias=False, kernel_initializer='he_uniform')(act)
                    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_transi)

            #BOTTLENECK
            if self.atrou:         
                block = DenseBlock(n_layers=self.nb_layers_per_block[-1],
                                   n_kernels=self.growth_rate,
                                   input_layer=x,
                                   activation_function=self.activation_function,
                                   concat_input=False,atrou=True,dilation_rate=2**(self.nb_scales-1))
            else:
                 block = DenseBlock(n_layers=self.nb_layers_per_block[-1],
                                   n_kernels=self.growth_rate,
                                   input_layer=x,
                                   activation_function=self.activation_function,
                                   concat_input=False)
           

            #EXPANSIVE PATH
            for b in range(self.nb_scales-2, -1, -1):
                if self.atrou:         
                    x = Concatenate(axis = 3)([block, skip_connect[b+offscale]])
                    block = DenseBlock(n_layers=self.nb_layers_per_block[b],
                                   n_kernels=self.growth_rate,input_layer=x, 
                                   activation_function=self.activation_function,
                                   atrou=True,dilation_rate=2**(b),concat_input=False)
                else:   
                    up = Conv2D(self.nb_layers_per_block[b+1]*self.growth_rate, 2, activation = self.activation_function,
                             padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(block))
                    x = Concatenate(axis = 3)([up, skip_connect[b+offscale]])
                    block = DenseBlock(n_layers=self.nb_layers_per_block[b],
                                   n_kernels=self.growth_rate,
                                   input_layer=x,
                                   activation_function=self.activation_function,
                                   concat_input=False)

            #FUSION AND SKIP CONNECT
            bn = BatchNormalization()(block)
            act = Activation(self.activation_function)(bn)
            conv = Conv2D(1, 1, activation = 'linear')(act)
            if self.resNet:
                output=Add()([conv,skip_connect[0]])
            else:
                output=conv
            #if self.shape_constraint :
            #outputs=Concatenate([output,window,norm])
            self.model = Model(input = [inputs,window,norm], outputs =output)
            self.model.compile(optimizer = Adam(lr=1e-3), loss = self.shape_loss, metrics=[metrics.mse,self.shape_metric])
           # else:
               # self.model = Model(input = inputs, outputs = output)
                #self.model.compile(optimizer = Adam(lr=1e-3), loss = 'mse') 

        else:
            print('Loadind model...')
            print(model_file)
            self.model = DeepNet.build_model(DeepNet,model_file=model_file)

        if verbose:
            print(self.model.summary())
