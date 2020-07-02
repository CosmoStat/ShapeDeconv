#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 09:34:24 2018

@author: alechat
"""
import os, sys
if os.path.dirname(os.path.dirname(os.path.realpath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from astropy.io import fits as fits
import numpy as np
import matplotlib.pylab as plt

from DeepDeconv.admm_radio import ADMM
from DeepDeconv.deepnet.DeepNet import DeepNet
from DeepDeconv.utils.conv_utils import get_conv2d, perform_shift_in_frequency
from DeepDeconv.utils.file_utils import fits2npy
from DeepDeconv.utils.deconv_utils import tikhonov
from DeepDeconv.utils.data_utils import add_noise, max_sv

def FISTA_radio(X, psf, init, rho, sigma, max_iter=500):
    M = np.zeros((193,193))
    M[47:47+96, 47:47+96] = 1
    X = M*X
    x_fista_old = init.copy()
    z_fista_old = init.copy()
    t_fista_old = 1
    alpha = 1. / ((max_sv(psf)+rho)*(1+1e-2)) # alpha is the gradient step
    
    ## Iterate
    for i in range(max_iter):
        grad = get_conv2d(M*get_conv2d(z_fista_old, psf, mode='scipy_fft', transp=False) - X, psf, mode='scipy_fft', transp=True)
        x_fista_new = z_fista_old - alpha * (grad + rho * sigma**2 * z_fista_old)
        x_fista_new[x_fista_new<0] = 0
        t_fista_new = (1. + np.sqrt(4. * t_fista_old**2 + 1.))/2.
        lambda_fista = 1 + (t_fista_old - 1)/t_fista_new
        z_fista_new = x_fista_old + lambda_fista * (x_fista_new - x_fista_old)
        np.copyto(x_fista_old, x_fista_new)
        np.copyto(z_fista_old, z_fista_new)
        t_fista_old = t_fista_new
    return x_fista_old

if __name__=='__main__':
      ## extra imports to set GPU options
      import tensorflow as tf
      from keras import backend as k
 

      psf_file = '/data/DeepDeconv/data/gauss_fwhm0p07/starfield_image-000-0.fits'
      model_file = '/data/DeepDeconv/model/DenoiserNet_vsc_noise0p04.hdf5'
      #model_file_tikhonet = '/data/DeepDeconv/model/DenseNet_vsc_rangeSNR20to100_nol2norm.hdf5'

      #NEURAL NETWORK
      if 'dnn' not in locals() and 'tikhonet' not in locals():
          ###################################
          # TensorFlow wizardry
          config = tf.ConfigProto()
          # Don't pre-allocate memory; allocate as-needed
          config.gpu_options.allow_growth = True
          # Only allow a total of half the GPU memory to be allocated
          config.gpu_options.per_process_gpu_memory_fraction = 0.5
          # Create a session with the above options specified.
          k.tensorflow_backend.set_session(tf.Session(config=config))
          dnn = DeepNet(network_name='ADMM net', model_file=model_file)
      #    tikhonet = DeepNet(network_name='Tikhonet', model_file=model_file_tikhonet)

      ### INPUT
      psf = fits.getdata('/data/DeepDeconv/data/Lofar-HBA.psf.fits')
      psf = psf.reshape(128,128)
      psf = psf[16:112,16:112]
      psf = psf/np.sum(psf)
      psfext = np.zeros((193,193))
      psfext[47:47+96,47:47+96] = psf

      idx = 6
      gal = fits2npy('/data/DeepDeconv/data/vsc_euclidpsfs/image-000-0.fits', [idx], 4)[0]
      galext = np.zeros((193,193))
      galext[47:47+96,47:47+96] = gal

      M = np.zeros((193,193))
      M[47:47+96, 47:47+96] = 1

      conv = get_conv2d(galext, psfext)


      noisy,_,sigma = add_noise(conv, std=np.max(conv)/10)
      noisy = M*noisy
      #noisy = conv
      sigma = sigma[0]
      #sigma = 0


      plt.figure(figsize=(18,4))
      plt.subplot(141)
      plt.imshow(psf)
      plt.title('PSF')
      plt.colorbar()
      plt.subplot(142)
      plt.imshow(gal)
      plt.title('Ground truth')
      plt.colorbar()
      plt.subplot(143)
      plt.imshow(conv)
      plt.title('Convolved')
      plt.colorbar()
      plt.subplot(144)
      plt.imshow(noisy)
      plt.title('Noisy')
      plt.colorbar()
      plt.show()

      #### TIKHONOV
      ##psf = psf[16:112,16:112]
      #tikho = tikhonov(noisy, psf, tau=1./20.)
      #tikho_dnn = tikhonet.model.predict(tikho.reshape((1,96,96,1)))


      ### FISTA
      init = np.zeros((193,193))
      #init = galext
      fi = FISTA_radio(noisy, psfext, init = init, rho=100, sigma=sigma, max_iter=1000)


      plt.figure()
      plt.imshow(fi)
      plt.colorbar()
      plt.title('FISTA')
      plt.show()

      ### ADMM
      #PARAMETERS ADMM
      max_iter = 20
      gamma = 1.4
      rho = 50
      rho_cap = 800
      eta = 0.5
      opti_method = 'FISTA' #'FISTA' 'GD'


      #INITIALISATION
      mu = np.zeros((193,193))

      x = np.zeros((193,193))
      z = np.zeros((193,193))


      #ADMM
      ad = ADMM(x, z, mu, noisy, psfext, sigma, dnn, rho=rho, gamma=gamma, eta=eta, rho_cap=rho_cap, mask=M)
      ad.iterate(max_iter=max_iter, plot_iter=1, verbose=False, plot_verbose=False)

      #FINAL PLOT
      result = ad.x_final
      convolved = get_conv2d(result, psfext).reshape(193,193)
      plt.figure(figsize=(17,4))
      plt.subplot(141)
      plt.imshow(result[47:47+96,47:47+96])
      plt.title('ADMM')
      plt.colorbar()
      plt.subplot(142)
      plt.imshow(result[47:47+96,47:47+96]-galext[47:47+96,47:47+96])
      plt.title('X-target')
      plt.colorbar()
      plt.subplot(143)
      plt.imshow(convolved[47:47+96,47:47+96])
      plt.title('X * PSF')
      plt.colorbar()
      plt.subplot(144)
      plt.imshow(noisy[47:47+96,47:47+96] - convolved[47:47+96,47:47+96])
      plt.title('Y - X * PSF')
      plt.colorbar()
      plt.show()

