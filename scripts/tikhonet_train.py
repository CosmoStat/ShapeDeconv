#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
SHAPE DECONVOLUTION DEEP NETWORK

This script builds and trains a UNET which goal is to denoise images that were
deconvolved using a Wiener filter.

:Authors: Fadi Nammour <fadi.nammour@cea.fr>
          François Lanusse <francois.lanusse@cea.fr>
          Hippolyte Karakostanoglou
"""

import os

# Dependency imports
from absl import flags, app
import numpy as np
from scipy import fft
import tensorflow as tf
import cadmos_lib as cl
import galflow as gf
from galaxy2galaxy import problems

# #flag example
# flags.DEFINE_bool(
#     "fake_data",
#     default=False,
#     help="If true, uses fake data instead of MNIST.")

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
    'n_shearlet', default=3, help="Number of shearlets scales.")

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

flags.DEFINE_float(
    'gamma', default=1., help="Trade-off parameter for shape constraint.")


FLAGS = flags.FLAGS

def ir2tf(imp_resp, shape):
    

    dim = 2
    # Zero padding and fill
    irpadded = np.zeros(shape)
    irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp
    # Roll for zero convention of the fft to avoid the phase
    # problem. Work with odd and even size.
    for axis, axis_size in enumerate(imp_resp.shape):

        irpadded = np.roll(irpadded,
                           shift=-int(np.floor(axis_size / 2)),
                           axis=axis)

    return fft.rfftn(irpadded, axes=range(-dim, 0))

def laplacian(shape):
    
    impr = np.zeros([3,3])
    for dim in range(2):
        idx = tuple([slice(1, 2)] * dim +
                    [slice(None)] +
                    [slice(1, 2)] * (1 - dim))
        impr[idx] = np.array([-1.0,
                              0.0,
                              -1.0]).reshape([-1 if i == dim else 1
                                              for i in range(2)])
    impr[(slice(1, 2), ) * 2] = 4.0
    return ir2tf(impr, shape), impr

def laplacian_tf(shape):
    return tf.convert_to_tensor(laplacian(shape)[0])

def wiener_tf(image, psf, balance):
    reg = laplacian_tf(image.shape)
    if psf.shape != reg.shape:
        trans_func = tf.signal.rfft2d(tf.signal.ifftshift(tf.cast(psf, 'float32')))
    else:
        trans_func = psf
    
    arg1 = tf.cast(tf.math.conj(trans_func), 'complex64')
    arg2 = tf.dtypes.cast(tf.math.abs(trans_func),'complex64') ** 2
    arg3 = balance * tf.dtypes.cast(tf.math.abs(laplacian_tf(image.shape)), 'complex64')**2
    wiener_filter = arg1 / (arg2 + arg3)
    
    wiener_applied = wiener_filter * tf.signal.rfft2d(tf.cast(image, 'float32'))
    
    return wiener_applied, trans_func

def pre_proc_unet(dico):
    r"""Preprocess the data and apply the Tikhonoff filter on the input galaxy images.

    This function takes the dictionnary of galaxy images and PSF for the input and
    the target and returns a list containing 2 arrays: an array of galaxy images that
    are the output of the Tikhonoff filter and an array of target galaxy images.

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
        Tikhonoff filter and an array of target galaxy images.
    
    Example
    -------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> from galaxy2galaxy import problems # to list avaible problems run problems.available()
    >>> problem128 = problems.problem('attrs2img_cosmos_hst2euclide')
    >>> dset = problem128.dataset(Modes.TRAIN, data_dir='attrs2img_cosmos_hst2euclide')
    >>> dset = dset.map(pre_proc_unet)
    """
    # Friest, we add noise
    # For the estimation of CFHT noise standard deviation check section 3 of:
    # https://github.com/CosmoStat/ShapeDeconv/blob/master/data/CFHT/HST2CFHT.ipynb
    sigma_cfht = 23.59
    noise = tf.random_normal(shape=tf.shape(dico['inputs']), mean=0.0, stddev=sigma_cfht, dtype=tf.float32)
    dico['inputs'] = dico['inputs'] + noise
    # Second, we interpolate the image on a finer grid
    x_interpolant=tf.image.ResizeMethod.BICUBIC
    interp_factor = 2
    Nx = 64
    Ny = 64
    dico['inputs_cfht'] = tf.image.resize(dico['inputs'],
                    [Nx*interp_factor,
                    Ny*interp_factor],
                    method=x_interpolant)
    # Since we lower the resolution of the image, we also scale the flux
    # accordingly
    dico['inputs_cfht'] = dico['inputs_cfht'] / interp_factor**2
    
    balance = 10**(-2.16)  #best after grid search
    dico['inputs_tikho'], dico['psf_cfht'] = wiener_tf(dico['inputs_cfht'][...,0], dico['psf_cfht'][...,0], balance)
    dico['inputs_tikho'] = tf.expand_dims(dico['inputs_tikho'], axis=0)
    dico['psf_cfht'] = tf.expand_dims(tf.cast(dico['psf_cfht'], 'complex64'), axis=0)
    dico['inputs_tikho'] = gf.kconvolve(dico['inputs_tikho'], dico['psf_cfht'],zero_padding_factor=1,interp_factor=interp_factor)
    dico['inputs_tikho'] = dico['inputs_tikho'][0,...]
    
    return dico['inputs_tikho'], dico['targets']


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

def single_window_metric(y_pred,y_true):
    r"""Compute the single window shape constraint value. By noting `y_pred` 
    and `y_true` as :math:`Y_p` and :math:`Y_t` the shape constraint is

    .. math:: \gamma sum_(i=1)^6 \mu_i \<W \odot (Y_t-Y_p),U_i \>^2

    Parameters
    ----------
    y_pred : Tensorflow tensor
        Batch of gray scale prediction images.
    y_true : Tensorflow tensor
        Batch of gray scale true images.

    Returns
    -------
    shape_constraint : float
        Single window shape constraint value.

    See Also
    --------
    multi_window_metric : the multi window version of the shape constraint.
    tikho_loss : loss function using the shape constraint.
    
    Notes
    -----
    The window and scalar weights required to compute the shape constraint are
    given as part of the input of the UNET.

    References
    ----------

    .. [1] F. Nammour, M. A. Schmitz, F. M. Ngolè Mboula, J. N. Girard, J.-L. 
       Starck, "Astronomical image restoration with shape constraint,"
       Journal of Fourier Analysis and Application, manuscript submitted for 
       publication, 2020.

    Example
    -------
    Using the single window shape constraint as a metric in a basic Neural
    Netowrk.
    
    >>> inputs = tf.keras.Input(shape=[64, 64, 1])
    >>> window = tf.keras.Input((64, 64,1),name='window')
    >>> weights = tf.keras.Input((6,1,1),name='weights')
    >>> net = tf.keras.layers.Conv2D(32, 3, padding='same')(inputs)
    >>> net = tf.keras.layers.Activation('relu')(net)
    >>> net = tf.keras.layers.Conv2D(16, 3, padding='same')(net)
    >>> outputs = tf.keras.layers.Conv2D(1, 3, padding='same')(net)
    >>> model = tf.keras.Model(inputs=[inputs,window,weights], outputs=outputs)
    >>> model.compile(optimizer = tf.keras.optimizers.Adam(lr=1e-3), 
                      loss = loss, weighted_metrics=[single_window_metric])
    """
    residual = y_true-y_pred
    window, mu = model.input[1:]
    shape_constraint=0
    for i in range(6):
        shape_constraint+=FLAGS.gamma*mu[i]*\
        tf.keras.backend.square(
            (tf.keras.backend.sum(residual*window*U_tensor[i],axis=(1,2,3))))/2
    return shape_constraint

def multi_window_metric(y_pred,y_true):
    r"""Compute the multi window shape constraint value. By noting `y_pred` 
    and `y_true` as :math:`Y_p` and :math:`Y_t` the shape constraint is

    .. math:: \gamma sum_(i=1)^6 \sum_{j=1}^J \mu_{ij} \<Y_t-Y_p,\Psi_j^\ast
              \(U_i\) \>^2

    Parameters
    ----------
    y_pred : Tensorflow tensor
        Batch of gray scale prediction images.
    y_true : Tensorflow tensor
        Batch of gray scale true images.

    Returns
    -------
    shape_constraint : float
        Multi window shape constraint value.

    See Also
    --------
    single_window_metric : the single window version of the shape constraint.
    tikho_loss : loss function using the shape constraint.

    References
    ----------

    .. [1] F. Nammour, M. A. Schmitz, F. M. Ngolè Mboula, J. N. Girard, J.-L. 
       Starck, "Astronomical image restoration with shape constraint,"
       Journal of Fourier Analysis and Application, manuscript submitted for 
       publication, 2020.

    Example
    -------
    Using the single window shape constraint as a metric in a basic Neural
    Netowrk.
    
    >>> inputs = tf.keras.Input(shape=[64, 64, 1])
    >>> net = tf.keras.layers.Conv2D(32, 3, padding='same')(inputs)
    >>> net = tf.keras.layers.Activation('relu')(net)
    >>> net = tf.keras.layers.Conv2D(16, 3, padding='same')(net)
    >>> outputs = tf.keras.layers.Conv2D(1, 3, padding='same')(net)
    >>> model = tf.keras.Model(inputs=inputs, outputs=outputs)
    >>> model.compile(optimizer = tf.keras.optimizers.Adam(lr=1e-3), 
                      loss = loss, weighted_metrics=[multi_window_metric])
    """
    residual=y_pred-y_true
    shape_constraint=0
    print("########################################################")
    print("########################################################")
    print("###################### GAMMA PSU_TENSOR SHAPES #################")
    print(psu.shape)
    print("########################################################")
    print("########################################################")
    for i in range(6):
        for j in range(psu.shape[1]):
            shape_constraint+=FLAGS.gamma*mu[i,j]*\
            tf.keras.backend.square(
                tf.keras.backend.sum(residual*psu_tensor[i,j],axis=(1,2,3)))/2.
    return shape_constraint

def tikho_loss(y_pred,y_true):
    r"""Compute the loss value of the Tikhonet. The value includes the MSE and 
    the shape constraint eventually. By noting the shape constraint as `M` 
    and `y_pred` and `y_true` as :math:`Y_p` and :math:`Y_t` the loss is

    .. math:: \|Y_t-Y_p\|^2 + \gamma M(Y_p)

    Parameters
    ----------
    y_pred : Tensorflow tensor
        Batch of gray scale prediction images.
    y_true : Tensorflow tensor
        Batch of gray scale true images.

    Returns
    -------
    data_fid+shape_constraint : float
        Tikhonet loss value.

    See Also
    --------
    single_window_metric : the single window version of the shape constraint.
    multi_window_metric : the multi window version of the shape constraint.

    References
    ----------

    .. [1] F. Nammour, M. A. Schmitz, F. M. Ngolè Mboula, J. N. Girard, J.-L. 
       Starck, "Astronomical image restoration with shape constraint,"
       Journal of Fourier Analysis and Application, manuscript submitted for 
       publication, 2020.

    Example
    -------
    Using `tikho_loss` as a custom loss in a basic Neural Netowrk.
    
    >>> inputs = tf.keras.Input(shape=[64, 64, 1])
    >>> net = tf.keras.layers.Conv2D(32, 3, padding='same')(inputs)
    >>> net = tf.keras.layers.Activation('relu')(net)
    >>> net = tf.keras.layers.Conv2D(16, 3, padding='same')(net)
    >>> outputs = tf.keras.layers.Conv2D(1, 3, padding='same')(net)
    >>> model = tf.keras.Model(inputs=inputs, outputs=outputs)
    >>> model.compile(optimizer = tf.keras.optimizers.Adam(lr=1e-3), 
                      loss = tikho_loss, weighted_metrics=[accuracy])
    """
    residual = y_true-y_pred
    data_fid = tf.keras.backend.sum(
        tf.keras.backend.square(residual),axis=(1,2,3))/2.
    if FLAGS.shape_constraint == 'None':
        shape_constraint = 0
    elif FLAGS.shape_constraint == 'single':
        shape_constraint = single_window_metric(y_pred,y_true)
    elif FLAGS.shape_constraint == 'multi':
        shape_constraint = multi_window_metric(y_pred, y_true)
    return data_fid+shape_constraint

def main(argv):
    
    if FLAGS.shape_constraint != 'None':
        U = cl.makeUi(FLAGS.n_row,FLAGS.n_col)
        if FLAGS.shape_constraint == 'single':
            U_tensor = tf.reshape(U, [U.shape[0],1,*U.shape[1:],1])
        if FLAGS.shape_constraint == 'multi':
            # Generate the adjoints of shearlets applied to U
            # get shearlets filters and their adjoints
            shearlets,shearlets_adj = cl.get_shearlets(FLAGS.n_row,FLAGS.n_col
                                                       ,FLAGS.n_shearlet)
            global mu, psu, psu_tensor
            # shealret adjoint of U, i.e Psi^{Star}(U)
            psu = np.array([cl.convolve_stack(ui,shearlets_adj) for ui in U])
            mu = cl.comp_mu(psu)
            mu = tf.reshape(mu, [*mu.shape])
            psu_tensor = tf.reshape(psu, [*psu.shape[:2],1,*psu.shape[2:],1])

    # DATA GENERATOR INITIALIZATION
    Modes = tf.estimator.ModeKeys
    problem128 = problems.problem('attrs2img_cosmos_cfht2hst')
    dset = problem128.dataset(Modes.TRAIN, data_dir=FLAGS.data_dir)
    dset = dset.repeat()
    dset = dset.map(pre_proc_unet)
    dset = dset.batch(FLAGS.batch_size)
    
    
    #NETWORK CONSTRUCTION STARTS HERE
    inputs = tf.keras.Input(shape=[FLAGS.n_row, FLAGS.n_col, 1]
                            , name='input_image')
    if FLAGS.shape_constraint=='single':
        window = tf.keras.Input((FLAGS.n_row, FLAGS.n_col,1),name='window')
        norm = tf.keras.Input((6,1,1),name='norm')
    
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
    
    if FLAGS.shape_constraint == 'single':
        model = tf.keras.Model(input = [inputs,window,norm], outputs=outputs)
        metrics += [single_window_metric]
    else:
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        if FLAGS.shape_constraint == 'multi':
            metrics += [multi_window_metric] 
    model.compile(optimizer = tf.keras.optimizers.Adam(lr=1e-3)
                  , loss = tikho_loss, weighted_metrics=metrics)
    # Train model
    history = model.fit(dset, steps_per_epoch=FLAGS.steps,epochs=FLAGS.epochs)
    
    # Save model
    model_name = 'tikhonet_{0}-constraint_scales-{1}'\
                  .format(FLAGS.shape_constraint,FLAGS.n_scale)
    if FLAGS.shape_constraint != 'None':
        model_name += '_gamma-{}'.format(FLAGS.gamma)
        if FLAGS.shape_constraint == 'multi':
            model += '_shearlet-{}'.format(FLAGS.n_shearlet)
    model_name += ('_steps-{0}_epochs-{1}_growth_rate-{2}_batch_size-{3}'+\
                   '_activationfunction-{4}').format(FLAGS.steps,FLAGS.epochs
                                                   ,FLAGS.growth_rate
                                                   ,FLAGS.batch_size
                                                   ,FLAGS.activation_function)
    model.save(FLAGS.model_dir+model_name)

if __name__ == '__main__':
  app.run(main)
