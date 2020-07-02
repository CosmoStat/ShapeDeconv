#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:53:50 2018

@author: alechat
"""
import os, sys
if os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    
from keras.models import Model, load_model
from keras.layers import Input, Concatenate, Conv2D, UpSampling2D, SeparableConv2D, BatchNormalization, AveragePooling2D, Activation
from keras.optimizers import Adam


from DeepDeconv.deepnet.DeepNet import DeepNet



class DeconvNet(DeepNet):
    
    def __init__(self, network_name = 'DeconvNet', img_rows = 96, img_cols = 96, model_file='', verbose=False):
        super(DeconvNet, self).__init__(network_name, img_rows, img_cols, model_file, verbose)
    
    def build_model(self, model_file = '', verbose = False):
        if model_file == '':
            inputs = Input((self.img_rows, self.img_cols, 1))
            
            #INPUT CONV
            conv_input = Conv2D(32, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(inputs)
            
            #DENSE BLOCK 1
            bn1 = BatchNormalization()(conv_input)
            act1 = Activation('swish')(bn1)
            conv1_1 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act1)
            
            bn1 = BatchNormalization()(conv1_1)
            act1 = Activation('swish')(bn1)
            conv1_2 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act1)
            
            x = Concatenate(axis = 3)([conv1_1, conv1_2])
            
            bn1 = BatchNormalization()(x)
            act1 = Activation('swish')(bn1)
            conv1_3 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act1)
            
            x = Concatenate(axis = 3)([conv1_1, conv1_2, conv1_3])
            
            bn1 = BatchNormalization()(x)
            act1 = Activation('swish')(bn1)
            conv1_4 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act1)
            
            x_1 = Concatenate(axis = 3)([conv_input, conv1_1, conv1_2, conv1_3, conv1_4])
            
            #POOLING LAYER 1
            bn1 = BatchNormalization()(x_1)
            act1 = Activation('swish')(bn1)
            conv_transi = Conv2D(80, 1, padding='same', use_bias=False, kernel_initializer='he_uniform')(act1)
            pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_transi)
            
            #DENSE BLOCK 2
            bn2 = BatchNormalization()(pool1)
            act2 = Activation('swish')(bn2)
            conv2_1 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act2)
            
            bn2 = BatchNormalization()(conv2_1)
            act2 = Activation('swish')(bn2)
            conv2_2 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act2)
            
            x = Concatenate(axis = 3)([conv2_1, conv2_2])

            bn2 = BatchNormalization()(x)
            act2 = Activation('swish')(bn2)
            conv2_3 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act2)
            
            x = Concatenate(axis = 3)([conv2_1, conv2_2, conv2_3])

            bn2 = BatchNormalization()(x)
            act2 = Activation('swish')(bn2)
            conv2_4 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act2)
            
            x = Concatenate(axis = 3)([conv2_1, conv2_2, conv2_3, conv2_4])
            
            bn2 = BatchNormalization()(x)
            act2 = Activation('swish')(bn2)
            conv2_5 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act2)
            
            x_2 = Concatenate(axis = 3)([pool1, conv2_1, conv2_2, conv2_3, conv2_4, conv2_5])
            
            #POOLING LAYER 2
            bn2 = BatchNormalization()(x_2)
            act2 = Activation('swish')(bn2)
            conv_transi = Conv2D(140, 1, padding='same', use_bias=False, kernel_initializer='he_uniform')(act2)
            pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_transi)

            #DENSE BLOCK 3
            bn3 = BatchNormalization()(pool2)
            act3 = Activation('swish')(bn3)
            conv3_1 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act3)
            
            bn3 = BatchNormalization()(conv3_1)
            act3 = Activation('swish')(bn3)
            conv3_2 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act3)
            
            x = Concatenate(axis = 3)([conv3_1, conv3_2])

            bn3 = BatchNormalization()(x)
            act3 = Activation('swish')(bn3)
            conv3_3 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act3)
            
            x = Concatenate(axis = 3)([conv3_1, conv3_2, conv3_3])

            bn3 = BatchNormalization()(x)
            act3 = Activation('swish')(bn3)
            conv3_4 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act3)
            
            x = Concatenate(axis = 3)([conv3_1, conv3_2, conv3_3, conv3_4]) 
            
            bn3 = BatchNormalization()(x)
            act3 = Activation('swish')(bn3)
            conv3_5 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act3)
            
            x = Concatenate(axis = 3)([conv3_1, conv3_2, conv3_3, conv3_4, conv3_5]) 
            
            bn3 = BatchNormalization()(x)
            act3 = Activation('swish')(bn3)
            conv3_6 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act3)
            
            x_3 = Concatenate(axis = 3)([pool2, conv3_1, conv3_2, conv3_3, conv3_4, conv3_5, conv3_6]) 
            
            #POOLING LAYER 3
            bn3 = BatchNormalization()(x_3)
            act3 = Activation('swish')(bn3)
            conv_transi = Conv2D(212, 1, padding='same', use_bias=False, kernel_initializer='he_uniform')(act3)
            pool3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_transi)

            #DENSE BLOCK 4 (BOTTLENECK)
            bn4 = BatchNormalization()(pool3)
            act4 = Activation('swish')(bn4)
            conv4_1 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act4)
            
            bn4 = BatchNormalization()(conv4_1)
            act4 = Activation('swish')(bn4)
            conv4_2 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act4)
            
            x = Concatenate(axis = 3)([conv4_1, conv4_2])

            bn4 = BatchNormalization()(x)
            act4 = Activation('swish')(bn4)
            conv4_3 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act4)
            
            x = Concatenate(axis = 3)([conv4_1, conv4_2, conv4_3])

            bn4 = BatchNormalization()(x)
            act4 = Activation('swish')(bn4)
            conv4_4 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act4)
            
            x = Concatenate(axis = 3)([conv4_1, conv4_2, conv4_3, conv4_4])
            
            bn4 = BatchNormalization()(x)
            act4 = Activation('swish')(bn4)
            conv4_5 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act4)
            
            x = Concatenate(axis = 3)([conv4_1, conv4_2, conv4_3, conv4_4, conv4_5])
            
            bn4 = BatchNormalization()(x)
            act4 = Activation('swish')(bn4)
            conv4_6 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act4)
            
            x = Concatenate(axis = 3)([conv4_1, conv4_2, conv4_3, conv4_4, conv4_5, conv4_6])
            
            bn4 = BatchNormalization()(x)
            act4 = Activation('swish')(bn4)
            conv4_7 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act4)
            
            x_4 = Concatenate(axis = 3)([conv4_1, conv4_2, conv4_3, conv4_4, conv4_5, conv4_6, conv4_7])
            
            #UPSAMPLING LAYER 1
            up1 = Conv2D(84, 2, activation = 'swish', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(x_4))
            x_up1 = Concatenate(axis = 3)([up1, x_3])
            
            #DENSE BLOCK 5
            bn5 = BatchNormalization()(x_up1)
            act5 = Activation('swish')(bn5)
            conv5_1 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act5)
            
            bn5 = BatchNormalization()(conv5_1)
            act5 = Activation('swish')(bn5)
            conv5_2 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act5)
            
            x = Concatenate(axis = 3)([conv5_1, conv5_2])

            bn5 = BatchNormalization()(x)
            act5 = Activation('swish')(bn5)
            conv5_3 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act5)
            
            x = Concatenate(axis = 3)([conv5_1, conv5_2, conv5_3])

            bn5 = BatchNormalization()(x)
            act5 = Activation('swish')(bn5)
            conv5_4 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act5)
            
            x = Concatenate(axis = 3)([conv5_1, conv5_2, conv5_3, conv5_4])

            bn5 = BatchNormalization()(x)
            act5 = Activation('swish')(bn5)
            conv5_5 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act5)
            
            x = Concatenate(axis = 3)([conv5_1, conv5_2, conv5_3, conv5_4, conv5_5])

            bn5 = BatchNormalization()(x)
            act5 = Activation('swish')(bn5)
            conv5_6 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act5)
            
            x_5 = Concatenate(axis = 3)([conv5_1, conv5_2, conv5_3, conv5_4, conv5_5, conv5_6])

            #UPSAMPLING LAYER 2
            up2 = Conv2D(72, 2, activation = 'swish', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(x_5))
            x_up2 = Concatenate(axis = 3)([up2, x_2])
            
            #DENSE BLOCK 6
            bn6 = BatchNormalization()(x_up2)
            act6 = Activation('swish')(bn6)
            conv6_1 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act6)
            
            bn6 = BatchNormalization()(conv6_1)
            act6 = Activation('swish')(bn6)
            conv6_2 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act6)
            
            x = Concatenate(axis = 3)([conv6_1, conv6_2])

            bn6 = BatchNormalization()(x)
            act6 = Activation('swish')(bn6)
            conv6_3 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act6)
            
            x = Concatenate(axis = 3)([conv6_1, conv6_2, conv6_3])

            bn6 = BatchNormalization()(x)
            act6 = Activation('swish')(bn6)
            conv6_4 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act6)
            
            x = Concatenate(axis = 3)([conv6_1, conv6_2, conv6_3, conv6_4])

            bn6 = BatchNormalization()(x)
            act6 = Activation('swish')(bn6)
            conv6_5 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act6)
                        
            x_6 = Concatenate(axis = 3)([conv6_1, conv6_2, conv6_3, conv6_4, conv6_5])

            #UPSAMPLING LAYER 3
            up3 = Conv2D(60, 2, activation = 'swish', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(x_6))
            x_up3 = Concatenate(axis = 3)([up3, x_1])
            
            #DENSE BLOCK 7
            bn7 = BatchNormalization()(x_up3)
            act7 = Activation('swish')(bn7)
            conv7_1 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act7)
            
            bn7 = BatchNormalization()(conv7_1)
            act7 = Activation('swish')(bn7)
            conv7_2 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act7)
            
            x = Concatenate(axis = 3)([conv7_1, conv7_2])

            bn7 = BatchNormalization()(x)
            act7 = Activation('swish')(bn7)
            conv7_3 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act7)
            
            x = Concatenate(axis = 3)([conv7_1, conv7_2, conv7_3])

            bn7 = BatchNormalization()(x)
            act7 = Activation('swish')(bn7)
            conv7_4 = SeparableConv2D(12, 3, padding='same', use_bias=False, kernel_initializer='he_uniform')(act7)
            
            x_7 = Concatenate(axis = 3)([conv7_1, conv7_2, conv7_3, conv7_4])

            #FUSION AND SKIP CONNECT
            bn8 = BatchNormalization()(x_7)
            act8 = Activation('swish')(bn8)
            conv8 = Conv2D(1, 1, activation = 'linear')(act8)

            self.model = Model(input = inputs, output = conv8)
            self.model.compile(optimizer = Adam(lr=1e-3), loss = 'mse')
            
        else:
            print('Loadind model...')
            print(model_file)
            self.model = load_model(model_file)
        
        if verbose:
            print(self.model.summary())
