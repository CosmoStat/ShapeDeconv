#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
SHAPE DECONVOLUTION DEEP NETWORK

This script builds and trains a UNET which goal is to denoise images.

:Authors: Fadi Nammour <fadi.nammour@cea.fr>
          François Lanusse <francois.lanusse@cea.fr>
          Utsav Akhaury
"""

import os

# Dependency imports
from absl import flags, app
import numpy as np
import tensorflow as tf
from galaxy2galaxy import problems

flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'model/'),
    help="Directory to put the model's fit.")

flags.DEFINE_string(
    'data_dir',
    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'data/'),
    help="Directory to read the data from.")

flags.DEFINE_string(
    'shape_constraint', default='None',
    help="Type of shape constraint used in the UNET architecture. It can be \
    either 'None', 'single' or 'multi'.")

flags.DEFINE_integer(
    'n_row', default=64, help="Number of rows per image.")

flags.DEFINE_integer(
    'n_col', default=64, help="Number of columns per image.")

flags.DEFINE_integer(
    'n_scale', default=4, help="Number of scales in the UNET.")

flags.DEFINE_integer(
    'steps', default=625, help="Number of steps per epoch in the training.")

flags.DEFINE_integer(
    'epochs', default=10, help="Number of epochs in the training.")

flags.DEFINE_integer(
    'growth_rate', default=12, help="Growth rate of the Dense Block.")

flags.DEFINE_integer(
    'batch_size', default=128, help="The batch size for training the UNET.")

flags.DEFINE_string(
    'activation_function',
    default='relu',
    help="Activation function used in the UNET.")


FLAGS = flags.FLAGS

def pre_proc_unet(dico):
    r"""Preprocess the data and add noise to the input galaxy images.

    This function takes the dictionnary of galaxy images and PSF for the input and
    the target and returns a list containing 2 arrays: an array of galaxy images that
    contains white additive Gaussian noise and an array of target galaxy images.

    Parameters
    ----------
    dico : dictionnary
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.  We can also refer to
        variables like `var1`.

    Returns
    -------
    list
        list containing 2 arrays: an array of galaxy images that are the output of the
        Tikhonov filter and an array of target galaxy images.

    Example
    -------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> from galaxy2galaxy import problems # to list avaible problems run problems.available()
    >>> problem128 = problems.problem('attrs2img_cosmos_hst2euclide')
    >>> dset = problem128.dataset(Modes.TRAIN, data_dir='attrs2img_cosmos_hst2euclide')
    >>> dset = dset.map(pre_proc_unet)
    """ 
    # add noise
    sigma = 7e-4
    # unifactor is a uniform radom variable factor to improve the training of the Unet
    unifactor = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
    noise = tf.random_normal(shape=tf.shape(dico['targets']), mean=0.0, stddev=sigma, dtype=tf.float32)
    dico['inputs'] = dico['targets'] + unifactor * noise
    
    # normalize the Unet inputs to improve the training
    norm_factor = 5e2
    dico['inputs'] = dico['inputs']*norm_factor
    dico['targets'] = dico['targets']*norm_factor
    
    return dico['inputs'], dico['targets']


def DenseBlock(n_layers, n_kernels, input_layer, activation_function='swish',
               axis_concat=3, concat_input=True):
    r"""
    Build a Dense Block connected to the input layer `input_layer`.
    Parameters
    ----------
    n_layers            : number of layers in the Dense Block.
    n_kernels           : number of convolution kernels per convolution layer.
    input_layers        : TensorFlow tensor given as an input layer for the
                          Dense Block.
    activation_function : function used for the activation layers. Default
                          `swish`.
    axis_concat         : Axis along which the layers are concatenated. 
                          Defalut 3.
    concat_input        : boolean to activate the skip concatenation of the 
                          input layer in the ouput layer of the Dense Block. 
                          Default `True`.
    Returns
    -------
    concat              : Tensorflow tensorflow. Output layer of the Dense 
                          Block.
    References
    ----------
    .. [1] Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017).
           *Densely connected convolutional networks*. In Proceedings of the 
           IEEE conference on computer vision and pattern recognition (pp. 
           4700-4708).
    Examples
    --------
    Build a dense block with a convolution layer as input layer.
    >>> inputs = tf.keras.Input(shape=[128, 128, 1])
    >>> denseblock = DenseBlock(n_layers=4, n_kernels=12, 
                                input_layer=inputs, activation_function='relu',
                                concat_input=True)
    """
    #concat_input: implies that we have a skip concatenation between input and 
    #output of block
    connect_input = input_layer
    for n in range(n_layers):
        bn = tf.keras.layers.BatchNormalization()(connect_input)
        act = tf.keras.layers.Activation(activation_function)(bn)
        conv = tf.keras.layers.SeparableConv2D(n_kernels, 3, padding='same', 
                                               use_bias=False, 
                                               kernel_initializer='he_uniform')\
                                              (act)
        if n == 0:
            concat = conv
        else:
            concat = tf.keras.layers.Concatenate(axis = 
                                                 axis_concat)([concat, conv])
        connect_input = concat
    if concat_input:
        return tf.keras.layers.Concatenate(axis = axis_concat)\
                                          ([input_layer, concat])
    return concat

def main(argv):

    # DATA GENERATOR INITIALIZATION
    Modes = tf.estimator.ModeKeys
    problem128 = problems.problem('meerkat_3600')
    dset = problem128.dataset(Modes.TRAIN, data_dir=FLAGS.data_dir)
    dset = dset.repeat()
    dset = dset.map(pre_proc_unet)
    dset = dset.batch(FLAGS.batch_size)
    
    
    #NETWORK CONSTRUCTION STARTS HERE
    inputs = tf.keras.Input(shape=[FLAGS.n_row, FLAGS.n_col, 1]
                            , name='input_image')
    
    #INPUT CONV
    net = tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False
                                 , kernel_initializer='he_uniform')(inputs)
    
    #CONTRACTING PATH
    nb_layers_per_block = [4,5,6,7]
    skip_connect = []
    
    for scale in range(FLAGS.n_scale-1):
        block = DenseBlock(n_layers=nb_layers_per_block[scale],
                       n_kernels=FLAGS.growth_rate,
                       input_layer=net,
                       activation_function=FLAGS.activation_function,
                       concat_input=True)
        skip_connect.append(block)
        batch_norm = tf.keras.layers.BatchNormalization()(block)
        activation = tf.keras.layers.Activation(FLAGS.activation_function)\
                                                (batch_norm)
        conv_transi = tf.keras.layers.Conv2D(32+np.sum(
                                                 nb_layers_per_block[:scale+1])
                                                *FLAGS.growth_rate, 1
                                            , padding='same'
                                            , use_bias=False
                                            , kernel_initializer='he_uniform')\
                                            (activation)
        net = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)
                                               , padding='same')(conv_transi)
    
    #BOTTLENECK
    block = DenseBlock(n_layers=nb_layers_per_block[-1],
                       n_kernels=FLAGS.growth_rate,
                       input_layer=net,
                       activation_function=FLAGS.activation_function,
                       concat_input=False)
    
    
    #EXPANSIVE PATH
    for scale in range(FLAGS.n_scale-2, -1, -1):   
        up = tf.keras.layers.Conv2D(nb_layers_per_block[scale+1]
                                    *FLAGS.growth_rate, 2
                                    , activation = FLAGS.activation_function
                                    , padding = 'same'
                                    , kernel_initializer = 'he_normal')\
                                    (tf.keras.layers.UpSampling2D(size = (2,2))
                                     (block))
        net = tf.keras.layers.Concatenate(axis = 3)([up, skip_connect[scale]])
        block = DenseBlock(n_layers=nb_layers_per_block[scale],
                       n_kernels=FLAGS.growth_rate,
                       input_layer=net,
                       activation_function=FLAGS.activation_function,
                       concat_input=False)
    
    #FUSION AND SKIP CONNECT
    batch_norm = tf.keras.layers.BatchNormalization()(block)
    activation = tf.keras.layers.Activation(FLAGS.activation_function)\
                                            (batch_norm)
    outputs = tf.keras.layers.Conv2D(1, 1, activation = 'linear')(activation)
    
    # Compile the model
    metrics = [tf.keras.metrics.mse,'accuracy']
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer = tf.keras.optimizers.Adam(lr=1e-3)
                  , loss = 'mse', weighted_metrics=metrics)
    # Train model
    history = model.fit(dset, steps_per_epoch=FLAGS.steps,epochs=FLAGS.epochs)
    
    # Save model
    model_name = ('unet_scales-{0}_steps-{1}_epochs-{2}_growth_rate-{3}_'+\
                   'batch_size-{4}_'+\
                  'activationfunction-{5}').format(FLAGS.n_scale
                                                   ,FLAGS.steps
                                                   ,FLAGS.epochs
                                                   ,FLAGS.growth_rate
                                                   ,FLAGS.batch_size
                                                   ,FLAGS.activation_function)
    model.save(FLAGS.model_dir+model_name)

if __name__ == '__main__':
  app.run(main)
