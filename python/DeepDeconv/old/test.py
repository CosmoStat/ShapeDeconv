#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 11:25:57 2018

@author: alechat
"""

if __name__=='__main__':
    import os, sys
    if os.path.dirname(os.path.dirname(os.path.realpath(__file__))) not in sys.path:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


    from DeepDeconv.deepnet.DeepNet import DeepNet
    from DeepDeconv.utils.batch_utils import get_batch_from_fits
    import numpy as np
    net_file = '/data/DeepDeconv/model/DenseNet_vsc_rangeSNR20to100_nol2norm.hdf5'
    if 'dnn' not in locals():
        dnn = DeepNet(model_file=net_file) #dnn is instance of DeepNet class
    # Input the file containing the galaxies and psfs for testing
    testset_file = '/data/DeepDeconv/data/vsc_euclidpsfs/image-000-0.fits'

    # Create the set of test with 10 observations at SNR 50
    if 'test_data' not in locals():
        test_data, target_data = get_batch_from_fits(testset_file, idx_list=np.arange(10), SNR=50,
                                noiseless_img_hdu=1, targets_hdu=4, psf_hdu=3,
                                image_dim=96, image_per_row=100,
                                deconv_mode='TIKHONOV')

    dnn_reconstruction = dnn.predict(test_data, verbose=1)
    print(dnn_reconstruction.shape)

    fm = dnn.get_layer_output(test_data, layer_idx=-4)
    print(fm.shape)

