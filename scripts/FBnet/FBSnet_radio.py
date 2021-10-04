#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding: utf-8

# Import Libraries
import sys
import pickle

# Define paths and directories
lib_path = '/gpfswork/rech/xdy/uze68md/GitHub/'
data_path = '/gpfswork/rech/xdy/uze68md/data/'
model_dir = '/gpfswork/rech/xdy/uze68md/trained_models/model_meerkat/'

# Add library path to PYTHONPATH
path_alphatransform = lib_path+'alpha-transform'
path_score = lib_path+'score'
sys.path.insert(0, path_alphatransform)
sys.path.insert(0, path_score)

# Load libraries
import numpy as np
import starlets
import tensorflow as tf
import cadmos_lib as cl

# Activate eager mode
tf.enable_eager_execution()


# Define function

def comp_grad(R,adj_U,mu,gamma):
    """This function returns the gradient of the differentiable part of the
    loss function.
    INPUT: R, 2D numpy array (residual)
           adj_U, 3D numpy array (adjoint shearlet transform of U)
           mu, 1D numpy array (weights associated to adj_U)
           gamma, scalar (trade-off between data-fidelity and shape constraint)
    OUTPUT: 2D numpy array"""
    temp = tf.zeros(R.shape,dtype=tf.dtypes.float32)
    for m,u in zip(mu,adj_U):
        for cst,im in zip(m,u):
            temp += cst*tf.keras.backend.sum(R*im)*im
    temp = gamma * temp + R
    return 2*temp

def convolve_tf(image, kernel):

    image = tf.expand_dims(tf.expand_dims(image, axis=0), axis=-1)
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
    result = tf.cast(tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME'), tf.float32)
    return tf.squeeze(result)

# Projection - Enforce non-negative values
def proj(xi):
    
    return tf.cast(tf.math.maximum(xi, 0.0), tf.float32)
 
    
# H operator
def H(data, psf):
    
    return convolve_tf(data, psf)


# H transpose operator
def Ht(data, psf):
    
    return convolve_tf(data, tf.reverse(tf.reverse(psf, axis=[0]), axis=[1]))      # rotate by 180


# The gradient
def grad(y, x_rec, psf):
    res = H(x_rec, psf) - y
    res = comp_grad(res,psu,mu,gamma)
    res = Ht(res, psf)
    return res


# Spectral value
def max_sv(psf):
    
    H = tf.signal.fft2d(tf.cast(psf, tf.complex64))
    normH = tf.math.abs(tf.reverse(tf.reverse(H, axis=[0]), axis=[1]) * H)
    return tf.cast(tf.math.reduce_max(normH), tf.float32)
     
    
# Compute gradient step size   
def get_alpha(sv):

    return (tf.cast(1.0, tf.float32) / 
            (sv * tf.cast(1.0 + 1.0e-5, tf.float32)))


def runFBS(y, x_0, psf, grad, n_iter, model):    
    
    # declare variables
    x_k = x_0
    
    sv = max_sv(psf)
    alpha = get_alpha(sv)

    for k in range(n_iter):
        
        ## Gradient Descent update  
        x_k1 = x_k - alpha * grad(y, x_k, psf)   
        
        # U-Net Denoising
        x_k1 = tf.expand_dims(tf.expand_dims(x_k1, axis=0), axis=-1)
        x_k1 = tf.cast(tf.squeeze(model(x_k1)), tf.float32)         
    
        # Update variables
        x_k = x_k1
    # Convert to numpy array
    x_k = tf.keras.backend.eval(x_k)     
    return x_k

# Load Data
f = open(data_path+"meerkat_batch.pkl", "rb")
batch = pickle.load(f)
f.close()
    
# SET SHAPE CONSTRAINT PARAMETERS

gamma = 0
n_batch,n_row,n_col = batch['psf'].shape
n_shearlet = 3

U = cl.makeUi(n_row,n_col)
_,shearlets_adj = cl.get_shearlets(n_row
                                   ,n_col
                                   ,n_shearlet)
#Adjoint shealret transform of U, i.e Psi^{Star}(U)
psu = np.array([cl.convolve_stack(ui,shearlets_adj) for ui in U])
mu = cl.comp_mu(psu)

# load unet denoiser
model_name = 'unet_scales-4_steps-6500_epochs-20_growth_rate-12_batch_size-32_activationfunction-relu'
model = tf.keras.models.load_model(model_dir + model_name)#, compile=False)

# set galaxies and psfs
gals_obs = batch['inputs']
psfs = batch['psf']
tikhos = batch['inputs_tikho']


# Run FBSnet
n_iter = 5

#loop
sol_g0 = []
i=1
for obs, psf, tikho in zip(gals_obs,psfs,tikhos):
    # Cast numpy arrays to tensors
    x_0 = tf.cast(tikho, tf.float32)
    gal_input_tf = tf.cast(obs, tf.float32)
    psf_tf = tf.cast(psf, tf.float32)
    #deconvolve
    recon = runFBS(gal_input_tf, x_0, psf_tf,
                   grad, n_iter, model)
    sol_g0 += [recon]
    if i%10 == 0:
        print(i)
    i += 1

filename = data_path + 'FBSnet_radio_g0'
np.save(filename,np.array(sol_g0))