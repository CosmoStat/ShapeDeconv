#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:08:20 2018

@author: alechat
"""
    
import numpy as np
from astropy.io import fits as fits
import matplotlib.pylab as plt
from numpy.fft import fft2, ifft2, fftshift
import time

from DeepDeconv.deepnet.DeepNet import DeepNet
from DeepDeconv.admm.admm import ADMM
from DeepDeconv.utils.conv_utils import get_conv2d, perform_shift_in_frequency
from DeepDeconv.utils.batch_utils import admm_data


## extra imports to set GPU options
import tensorflow as tf
from keras import backend as k
 

if __name__=='__main__':
    #FILES
    gal_file = '/data/DeepDeconv/data/csc_euclidpsfs/image-000-0.fits'
    psf_file = '/data/DeepDeconv/data/gauss_fwhm0p07/starfield_image-000-0.fits'
    model_file = '/data/DeepDeconv/model/DenoiserNet_csc_rangeSNR20to100.hdf5'
#    model_file_tikhonet = '/data/DeepDeconv/model/DenseNet_vsc_rangeSNR20to100_nol2norm.hdf5'
    
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

    for rho_cap in [400,800,200]:
        #PARAMETERS ADMM
        max_iter = 20
        gamma = 1.4
        rho = 50
    #    rho_cap = 800
        eta = 0.5
        opti_method = 'FISTA' #'FISTA' 'GD'
        
        for SNR in [20,60,100]:
            #PARAMETERS DATA
            noise_std = None
    #        SNR = 100
            first_index = 0
            nb_gal = 1024
            
        #        idx_list = [1004, 4026, 6044, 9004]
            idx_list = np.arange(first_index, first_index+nb_gal)
            
            
            #INSTANCIATE RESULTS
            result = np.zeros((len(idx_list), 96, 96))
            result_z = np.zeros((len(idx_list), 96, 96))
            psf0p07 = fits.getdata(psf_file)
            out_file ='/data/DeepDeconv/output_admm/admm_CSC_%d_SNR%d_rhocap%d_gamma1p4_denoisenet0p04.npy'%(nb_gal, SNR, rho_cap)
            print(out_file)
            
            c = 0
            for i in idx_list:
                print('PROCESSING GALAXY #%d'%(i))
                t1 = time.time()
                
                #LOAD DATA
                data, target, psf, snr_img, noise_std_output = admm_data(gal_file, i, 
                                                                         noise_std=noise_std, SNR=SNR,
                                                                         noiseless_img_hdu=1, targets_hdu=4, psf_hdu=3,
                                                                         image_dim=96, image_per_row=100)
               
                #ADD GAUSSIAN PSF
                psf_gauss = fits.getdata(psf_file)
                fft_gauss = fft2(psf_gauss)
                fft = fft2(psf)
                r = fft/fft_gauss
                r[np.abs(fft_gauss)<1e-6] = 0
                r = perform_shift_in_frequency(r, (96,96), np.array([1,1]))
                psf = np.abs(fftshift(ifft2(r)))
                
                #INITIALISATION
                mu = np.zeros((96,96))
                x = np.zeros((96,96))
                z = np.zeros((96,96))
                
                #ADMM
                ad = ADMM(x, z, mu, data, psf, noise_std_output, dnn, rho=rho, gamma=gamma, eta=eta,
                          opti_method=opti_method, rho_cap=rho_cap)
                ad.iterate(max_iter=max_iter, plot_iter=1, verbose=False, plot_verbose=False)
                
                #FINAL PLOT
                result[c] = ad.x_final
                result_z[c] = ad.z_final
                convolved = get_conv2d(result[c], psf).reshape(96,96)
                plt.figure(figsize=(17,10))
                plt.subplot(231)
                plt.imshow(data[28:69, 28:69])
                plt.title('Input')
                plt.colorbar()
                plt.subplot(232)
                plt.imshow(target[28:69, 28:69])
                plt.title('Target')
                plt.colorbar()
                plt.subplot(233)
                plt.imshow(result[c][28:69, 28:69])
                plt.title('X')
                plt.colorbar()
                plt.subplot(234)
                plt.imshow(result[c][28:69, 28:69]-target[28:69, 28:69])
                plt.title('X-target')
                plt.colorbar()
                plt.subplot(235)
                plt.imshow(convolved[28:69, 28:69])
                plt.title('X * PSF')
                plt.colorbar()
                plt.subplot(236)
                plt.imshow(data[28:69, 28:69] - convolved[28:69, 28:69])
                plt.title('Y - X * PSF')
                plt.colorbar()
                plt.show()
                print('Run time: %.1f s'%(time.time()-t1))
                
                c+=1
                
            #SAVE RESULT
            np.save(out_file, result)
        
        
