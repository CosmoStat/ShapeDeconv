#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 09:23:54 2018

@author: alechat
"""

import glob

## import the architecture to be trained
from DeepDeconv.deepnet.DeconvNet import DeconvNet as DNN

if __name__ == '__main__':
    gal_files = glob.glob('/data/DeepDeconv/data/vsc_euclidpsfs/image-0*.fits')
    gal_files.sort()
    validation_file = gal_files[0]
    train_files = gal_files[1:2]
    print(gal_files)
    noiseless_obs_hdu = 1
    targets_hdu = 4
    psf_hdu = 3
    
    batch_size = 32
    deconv_mode = 'TIKHONOV'
    model_file = '/data/DeepDeconv/model/saved_model.hdf5'
    noise_std = 0.04
    SNR = [20, 100]
    
    logfile = 'log.txt'
    

    net = DNN(network_name='DeconvNet', verbose=True)
    net.train_generator(train_files, validation_file, epochs=1, batch_size=batch_size, model_file = model_file,
                        nb_img_per_file=10000, validation_set_size=10000,
                        noise_std=noise_std, SNR=SNR,
                        noiseless_obs_hdu=noiseless_obs_hdu, targets_hdu=targets_hdu, psf_hdu=psf_hdu,
                        image_dim=96, image_per_row=100,
                        deconv_mode=None, rho_fista=1e-3,
                        logfile=logfile)
