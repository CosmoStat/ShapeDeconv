#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alechat/fsureau


Tools to pre-process and generate batches ready to use inside the deep learning and deconvolution methods.
"""

import os, sys
if os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import numpy as np
import random

from DeepDeconv.utils.file_utils import fits2npy,save2fits,stampCollection2Mosaic
from DeepDeconv.utils.data_utils import add_noise,mad_est_white_noise_list
from DeepDeconv.utils.deconv_utils_FCS import tikhonov_hyp,FISTA, tikhonov
from DeepDeconv.utils.conv_utils import tf_lap,tf_dirac
from astropy.io import fits as fits

def format_dataset(filename,
                   noiseless_img_hdu=0,
                   targets_hdu=0, psf_hdu=0,
                   idx_list=[],
                   image_dim=96,
                   image_per_row=100):
    """Read a fits file and format it into 3 numpy arrays for the noiseless images, the targets and the PSFs
    
    :param filename: path to the fits file
    :param noiseless_img_hdu: hdu corresponding to the noiseless image
    :param targets_hdu: hdu corresponding to the target image
    :param psf_hdu: hdu corresponding to the psf image
    :param idx_list: indices of the image to extract
    :param image_dim: Size of one side of one square image composing the mosaic.
    :param image_per_row: Number of images in one row of the mosaic (assumed square).
    :type filename: string
    :type noiseless_img_hdu: int
    :type targets_hdu: int
    :type psf_hdu: int
    :type idx_list: list of int
    :type image_dim: int
    :type image_per_row: int

    :returns: list of noiseless patches, targets and psfs 
    :rtype: list of 2D npy.ndarray,list of 2D npy.ndarray,list of 2D npy.ndarray
    
    .. note::
        This function calls :py:func:`DeepDeconv.utils.file_utils.fits2npy` three times.
    """
    noiseless_img = fits2npy(filename, idx_list, noiseless_img_hdu, image_dim=image_dim, image_per_row=image_per_row)
    targets = fits2npy(filename, idx_list, targets_hdu, image_dim=image_dim, image_per_row=image_per_row)
    if psf_hdu is None:
        return noiseless_img, targets
    psfs = fits2npy(filename, idx_list, psf_hdu, image_dim=image_dim, image_per_row=image_per_row)
    return noiseless_img, targets, psfs

def shuffle_new_fits(file_list, nb_img_per_file=10000,
                        noiseless_img_hdu=0, targets_hdu=0, psf_hdu=0,
                        image_dim=96, image_per_row=100,rootname='image-shfl'):
    """Reshuffle (permutation) a list of FITS file and save the results into FITS files. 
    
    :param file_list: list of paths to the fits file
    :param nb_img_per_file: number of patches in each (input and output) fits file
    :param noiseless_img_hdu: hdu corresponding to the noiseless image
    :param targets_hdu: hdu corresponding to the target image
    :param psf_hdu: hdu corresponding to the psf image
    :param image_dim: Size of one side of one square image composing the mosaic.
    :param image_per_row: Number of images in one row of the mosaic (assumed square).
    :param rootname: root of the FITS filename where images will be saved
    :type file_list: list of strings
    :type nb_img_per_file: int
    :type noiseless_img_hdu: int
    :type targets_hdu: int
    :type psf_hdu: int
    :type image_dim: int
    :type image_per_row: int
    :type rootname: string
    """
    
    nfiles=len(file_list)
    gal_shuffle = np.random.permutation(nb_img_per_file*nfiles)
    np.savetxt(rootname+"_galshuffle.txt",gal_shuffle)
    start_wr=0
    end_wr=nb_img_per_file
    for kfile_wr in range(nfiles):
        print("Process file {0}".format(kfile_wr))
        cur_gal_shuffle=gal_shuffle[start_wr:end_wr]
        filenb=cur_gal_shuffle//nb_img_per_file
        galnb=cur_gal_shuffle%nb_img_per_file
        np.savetxt(rootname+"_{0}_filenb.txt".format(kfile_wr),filenb)
        np.savetxt(rootname+"_{0}_galnb.txt".format(kfile_wr),galnb)
        start_rd=0
        image_array=np.zeros((nb_img_per_file,image_dim,image_dim))
        target_array=np.zeros((nb_img_per_file,image_dim,image_dim))
        psf_array=np.zeros((nb_img_per_file,image_dim,image_dim))
        for kfile_rd in range(nfiles):
            print("Process subfile {0}".format(kfile_rd))
            lst_rd=(np.where(filenb==kfile_rd))[0]
            if(len(lst_rd)>0):
                gal_idx=galnb[lst_rd]
                data_bundle = format_dataset(file_list[kfile_rd],noiseless_img_hdu=noiseless_img_hdu,
                                         targets_hdu=targets_hdu, psf_hdu=psf_hdu,idx_list=gal_idx,
                                         image_dim=image_dim,image_per_row=image_per_row)
                image_array[lst_rd]=data_bundle[0]
                target_array[lst_rd]=data_bundle[1]
                psf_array[lst_rd]=data_bundle[2]
        #save2fits(image_array,rootname+"-{0}-image.fits".format(kfile_wr), image_dim=image_dim,
        #          image_per_row=image_per_row, image_per_col=image_per_row)
        #save2fits(psf_array,rootname+"-{0}-psf.fits".format(kfile_wr), image_dim=image_dim,
        #          image_per_row=image_per_row, image_per_col=image_per_row)
        #save2fits(target_array,rootname+"-{0}-target.fits".format(kfile_wr), image_dim=image_dim,
        #          image_per_row=image_per_row, image_per_col=image_per_row)
        image_mosaic=stampCollection2Mosaic(image_array,image_dim=image_dim,image_per_row=image_per_row)
        psf_mosaic=stampCollection2Mosaic(psf_array,image_dim=image_dim,image_per_row=image_per_row)
        target_mosaic=stampCollection2Mosaic(target_array,image_dim=image_dim,image_per_row=image_per_row)
        hdu0=fits.PrimaryHDU(image_mosaic)
        hdu1=fits.ImageHDU(psf_mosaic)
        hdu2=fits.ImageHDU(target_mosaic)
        hdul=fits.HDUList([hdu0,hdu1,hdu2])
        hdul.writeto(rootname+"-{0}-multihdu.fits".format(kfile_wr),overwrite=True)
        start_wr=end_wr
        end_wr+=nb_img_per_file

        
def get_batch_noise_from_fits(filename, idx_list=np.arange(10000), noise_std=None, SNR=None,
                        noiseless_img_hdu=0, targets_hdu=0, psf_hdu=0,
                        image_dim=96, image_per_row=100,
                        deconv_mode=None, rho_fista=1e-3,
                        risktype="GCV",reg="Dirac",reg_frac=1,tol=1e-12):
    """Read a fits file and generate one batch of noise ready to feed to a neural network.
    
    :param filename: path to the fits file
    :param idx_list: indices of the image to extract
    :param noise_std: white noise standard deviation or range of standard deviation to add (superseded by SNR)
    :param SNR: desired output SNR (measured as :math:`SNR=\\frac{\\|X\\|_2}{\\sigma}`)
    :param noiseless_img_hdu: hdu corresponding to the noiseless image
    :param targets_hdu: hdu corresponding to the target image
    :param psf_hdu: hdu corresponding to the psf image
    :param image_dim: Size of one side of one square image composing the mosaic.
    :param image_per_row: Number of images in one row of the mosaic (assumed square).
    :param deconv_mode: whether None (no deconvolution), 'FISTA' (:py:func:`DeepDeconv.utils.deconv_utils_FCS.FISTA`), 
           'TIKHONOV' (:py:func:`DeepDeconv.utils.deconv_utils_FCS.tikhonov`) or 'TIKHONOV_HYP'(:py:func:`DeepDeconv.utils.deconv_utils_FCS.tikhonov_hyp`)
    :param rho_fista: augmented lagrangian hyperparameter
    :param risktype: risk to use in case of estimation of hyperparameters (TIKHONOV_HYP): whether 'GCV' (generalized cross-validation), 'SurePred' (SURE prediction risk min), 
           'SureProj' (SURE projection risk min) or 'PEREYRA'(hierarchical bayesian with gamma prior)     
    :param reg: regularization kernel (either "Dirac" or "Lap")
    :param reg_frac: fraction of selected hyperparameter to take for deconvolution
    :param tol: threshold to set orthogonal of null space in "SureProj"
    :type filename: string
    :type idx_list: list of int
    :type noise_std: float or [float,float]
    :type SNR: float or [float,float]
    :type noiseless_img_hdu: int
    :type targets_hdu: int
    :type psf_hdu: int
    :type image_dim: int
    :type image_per_row: int
    :type deconv_mode: string
    :type rho_fista: float
    :type risktype: string
    :type reg: string
    :type reg_frac: double (default: 1)
    :type tol: float

    :returns: array of processed images, array of noise only image
    :rtype: list of 2D npy.ndarray,list of 2D npy.ndarray
    
    .. note::
        The DNN works with batches with 4 dimensions (number of images, shape of image, shape of image, number of channels).
    """

    if deconv_mode is None:
        noiseless_img, targets = format_dataset(filename,
                                                noiseless_img_hdu=noiseless_img_hdu,
                                                targets_hdu=targets_hdu, psf_hdu=None,
                                                idx_list=idx_list,
                                                image_dim=image_dim,
                                                image_per_row=image_per_row)
    else:
        noiseless_img, targets, psfs = format_dataset(filename,
                                                      noiseless_img_hdu=noiseless_img_hdu,
                                                      targets_hdu=targets_hdu, psf_hdu=psf_hdu,
                                                      idx_list=idx_list,
                                                      image_dim=image_dim,
                                                      image_per_row=image_per_row)
    noisy_img, SNR_list, _ = add_noise(noiseless_img, std=noise_std, SNR=SNR)
    noisy_img-=noiseless_img
    if deconv_mode is None:
        return (noisy_img.reshape(len(idx_list), noisy_img[0].shape[0], noisy_img[0].shape[1], 1),
                targets.reshape(len(idx_list), targets[0].shape[0], targets[0].shape[1], 1))
    elif deconv_mode == 'TIKHONOV':
        deconv_img = tikhonov(noisy_img, psfs, 1./SNR_list)
    elif deconv_mode == 'FISTA':
        deconv_img = FISTA(noisy_img, psfs, 1e-3)
    elif deconv_mode == 'TIKHONOV_HYP':
        #Start by estimating noise
        var_est=mad_est_white_noise_list(noisy_img,wavelet_name='db3')**2
        if(reg=="Lap"):
            trans_reg,_=tf_lap(2,noisy_img[0].shape,is_real=False)
        else:
            trans_reg,_=tf_dirac(2,noisy_img[0].shape,is_real=False) 
        #Estimate hyperparameter and deconvolved image
        deconv_img,hyp=tikhonov_hyp(noisy_img,psfs,trans_reg,var_est,risktype=risktype,tol=tol,reg_frac=reg_frac)
    elif deconv_mode == 'FISTA_HYP':
        deconv_img = FISTA(noisy_img, psfs, 1e-3)
    else:
        ValueError('Invalid value for deconv_mode, correct inputs are \'TIKHONOV\', \'FISTA\' or None')
    return (deconv_img.reshape(len(idx_list), deconv_img[0].shape[0], deconv_img[0].shape[1], 1),
                noisy_img.reshape(len(idx_list), noisy_img[0].shape[0], noisy_img[0].shape[1], 1))


        
def get_batch_from_fits(filename, idx_list=np.arange(10000), noise_std=None, SNR=None,
                        noiseless_img_hdu=0, targets_hdu=0, psf_hdu=0,
                        image_dim=96, image_per_row=100,
                        deconv_mode=None, rho_fista=1e-3, return_noisy=False,
                        risktype="GCV",reg="Dirac",reg_frac=1.0,tol=1e-12,shape_constraint=False,
                        win_filename= None,win_hdu=0, mom_hdu=1,shearlet=False):
    """Read a fits file and generate one batch of data ready to feed to a neural network.
    
    :param filename: path to the fits file
    :param idx_list: indices of the image to extract
    :param noise_std: white noise standard deviation or range of standard deviation to add (superseded by SNR)
    :param SNR: desired output SNR (measured as :math:`SNR=\\frac{\\|X\\|_2}{\\sigma}`)
    :param noiseless_img_hdu: hdu corresponding to the noiseless image
    :param targets_hdu: hdu corresponding to the target image
    :param psf_hdu: hdu corresponding to the psf image
    :param image_dim: Size of one side of one square image composing the mosaic.
    :param image_per_row: Number of images in one row of the mosaic (assumed square).
    :param deconv_mode: whether None (no deconvolution), 'FISTA' (:py:func:`DeepDeconv.utils.deconv_utils_FCS.FISTA`), 
           'TIKHONOV' (:py:func:`DeepDeconv.utils.deconv_utils_FCS.tikhonov`) or 'TIKHONOV_HYP'(:py:func:`DeepDeconv.utils.deconv_utils_FCS.tikhonov_hyp`)
    :param rho_fista: augmented lagrangian hyperparameter
    :param return_noisy: whether to also return the noisy convolved image when deconv_mode is not None
    :param risktype: risk to use in case of estimation of hyperparameters (TIKHONOV_HYP): whether 'GCV' (generalized cross-validation), 'SurePred' (SURE prediction risk min), 
           'SureProj' (SURE projection risk min) or 'PEREYRA'(hierarchical bayesian with gamma prior)     
    :param reg: regularization kernel (either "Dirac" or "Lap")
    :param reg_frac: fraction of selected hyperparameter to take for deconvolution
    :param tol: threshold to set orthogonal of null space in "SureProj"
    :type filename: string
    :type idx_list: list of int
    :type noise_std: float or [float,float]
    :type SNR: float or [float,float]
    :type noiseless_img_hdu: int
    :type targets_hdu: int
    :type psf_hdu: int
    :type image_dim: int
    :type image_per_row: int
    :type deconv_mode: string
    :type rho_fista: float
    :type return_noisy: bool
    :type risktype: string
    :type reg: string
    :type reg_frac: double (default: 1)
    :type tol: float

    :returns: list of processed 2D images, list of target 2D images, list of noisy convolved 2D images (if return_noisy set and deconv_mode not None)
    :rtype: list of 2D npy.ndarray,list of 2D npy.ndarray,list of 2D npy.ndarray
    
    .. note::
        The DNN works with batches with 4 dimensions (number of images, shape of image, shape of image, number of channels).
    """
    if shape_constraint:
        batch_nowin=get_batch_from_fits(filename, idx_list=idx_list, noise_std=noise_std, SNR=SNR,
                        noiseless_img_hdu=noiseless_img_hdu, targets_hdu=targets_hdu, psf_hdu=psf_hdu,
                        image_dim=image_dim, image_per_row=image_per_row,
                        deconv_mode=deconv_mode, rho_fista=rho_fista, return_noisy=return_noisy,
                        risktype=risktype,reg=reg,reg_frac=reg_frac,
                        tol=tol,shape_constraint=False,win_filename=
                        None,win_hdu=0,mom_hdu=1)
        win_list=fits2npy(win_filename,idx_list, win_hdu)
        lns=tuple(list(np.shape(win_list))+[1])
        win_list=np.reshape(win_list,lns)
        norm_list=fits.getdata(win_filename, mom_hdu)[idx_list,:]
        lns=tuple(list(np.shape(norm_list))+[1]+[1])
        norm_list=np.reshape(norm_list,lns)
        if return_noisy:
            output=([batch_nowin[0], win_list,norm_list], batch_nowin[1],batch_nowin[2],batch_nowin[3])
        else:
            output=([batch_nowin[0], win_list,norm_list], batch_nowin[1],batch_nowin[2])
        return output
    if deconv_mode is None:
        noiseless_img, targets = format_dataset(filename,
                                                noiseless_img_hdu=noiseless_img_hdu,
                                                targets_hdu=targets_hdu, psf_hdu=None,
                                                idx_list=idx_list,
                                                image_dim=image_dim,
                                                image_per_row=image_per_row)
    else:
        noiseless_img, targets, psfs = format_dataset(filename,
                                                      noiseless_img_hdu=noiseless_img_hdu,
                                                      targets_hdu=targets_hdu, psf_hdu=psf_hdu,
                                                      idx_list=idx_list,
                                                      image_dim=image_dim,
                                                      image_per_row=image_per_row)
    noisy_img, SNR_list, std_list = add_noise(noiseless_img, std=noise_std, SNR=SNR)
    if deconv_mode is None:
        return (noisy_img.reshape(len(idx_list), noisy_img[0].shape[0], noisy_img[0].shape[1], 1),
                targets.reshape(len(idx_list), targets[0].shape[0], targets[0].shape[1], 1))
    elif deconv_mode == 'TIKHONOV':
        deconv_img = tikhonov(noisy_img, psfs, 1./SNR_list)
    elif deconv_mode == 'FISTA':
        deconv_img = FISTA(noisy_img, psfs, 1e-3)
    elif deconv_mode == 'TIKHONOV_HYP':
        #Start by estimating noise
        var_est=mad_est_white_noise_list(noisy_img,wavelet_name='db3')**2
        if(reg=="Lap"):
            trans_reg,_=tf_lap(2,noisy_img[0].shape,is_real=False)
        else:
            trans_reg,_=tf_dirac(2,noisy_img[0].shape,is_real=False) 
        #Estimate hyperparameter and deconvolved image
        deconv_img,hyp=tikhonov_hyp(noisy_img,psfs,trans_reg,var_est,risktype=risktype,tol=tol,reg_frac=reg_frac)
    elif deconv_mode == 'FISTA_HYP':
        deconv_img = FISTA(noisy_img, psfs, 1e-3)
    else:
        ValueError('Invalid value for deconv_mode, correct inputs are \'TIKHONOV\', \'FISTA\' or None')
    if return_noisy:
        return (deconv_img.reshape(len(idx_list), deconv_img[0].shape[0], deconv_img[0].shape[1], 1),
                targets.reshape(len(idx_list), targets[0].shape[0], targets[0].shape[1], 1),1./(std_list**2),
                noisy_img.reshape(len(idx_list), noisy_img[0].shape[0], noisy_img[0].shape[1], 1))
    else:
        return (deconv_img.reshape(len(idx_list), deconv_img[0].shape[0], deconv_img[0].shape[1], 1),
                targets.reshape(len(idx_list), targets[0].shape[0], targets[0].shape[1], 1),1./(std_list**2))



def dynamic_batches(fits_files, batch_size=100, nb_img_per_file=10000,
                        noise_std=None, SNR=None,
                        noiseless_img_hdu=0, targets_hdu=0, psf_hdu=0,
                        image_dim=96, image_per_row=100,
                        deconv_mode=None, rho_fista=1e-3,
                        risktype="GCV",reg="Dirac",reg_frac=1.0,tol=1e-12, 
                        shape_constraint=False,win_filename=None,win_hdu=0, mom_hdu=1):
    """Generator function used by the training function fit_generator in Keras.
    It is a function with an infinite loop generating one batch at a time (yield).
    The fit_generator process will call this function by itself to get only the number of batches needed without overloading the memory.
    
    :param fits_files: path to the FITS files
    :param batch_size: number of images in one batch.
    :param nb_img_per_file: number of images in FITS file.
    :param noise_std: white noise standard deviation or range of standard deviation to add (superseded by SNR)
    :param SNR: desired output SNR (measured as :math:`SNR=\\frac{\\|X\\|_2}{\\sigma}`)
    :param noiseless_img_hdu: hdu corresponding to the noiseless image
    :param targets_hdu: hdu corresponding to the target image
    :param psf_hdu: hdu corresponding to the psf image
    :param image_dim: Size of one side of one square image composing the mosaic.
    :param image_per_row: Number of images in one row of the mosaic (assumed square).
    :param deconv_mode: whether None (no deconvolution), 'FISTA' (:py:func:`DeepDeconv.utils.deconv_utils_FCS.FISTA`), 
           'TIKHONOV' (:py:func:`DeepDeconv.utils.deconv_utils_FCS.tikhonov`) or 'TIKHONOV_HYP'(:py:func:`DeepDeconv.utils.deconv_utils_FCS.tikhonov_hyp`)
    :param rho_fista: augmented lagrangian hyperparameter
    :param return_noisy: whether to also return the noisy convolved image when deconv_mode is not None
    :param risktype: risk to use in case of estimation of hyperparameters (TIKHONOV_HYP): whether 'GCV' (generalized cross-validation), 'SurePred' (SURE prediction risk min), 
           'SureProj' (SURE projection risk min) or 'PEREYRA'(hierarchical bayesian with gamma prior)     
    :param reg: regularization kernel (either "Dirac" or "Lap")
    :param reg_frac: fraction of selected hyperparameter to take for deconvolution
    :param tol: threshold to set orthogonal of null space in "SureProj"
    :type fits_files: list of string
    :type batch_size: int
    :type nb_img_per_file: int
    :type noise_std: float or [float,float]
    :type SNR: float or [float,float]
    :type noiseless_img_hdu: int
    :type targets_hdu: int
    :type psf_hdu: int
    :type image_dim: int
    :type image_per_row: int
    :type deconv_mode: string
    :type rho_fista: float
    :type return_noisy: bool
    :type risktype: string
    :type reg: string
    :type reg_frac: double (default: 1)
    :type tol: float

    :returns: yields list of processed images, targets
    :rtype: list of 2D npy.ndarray,list of 2D npy.ndarray
    
    """
    counter = 0
    while True:
        if counter == 0:
            random.shuffle(fits_files)
        filename = fits_files[counter]
        if win_filename is not None:
            winname=win_filename[counter]
        else :
            winname=None
        counter = (counter + 1) % len(fits_files) #LOOPING ON COUNTER
        gal_shuffle = np.random.permutation(nb_img_per_file)
        #NOTE the cbatch assumes nb_img_per_file to be 10000. Not correct and I changed it FCS.
        for cbatch in range(0,  nb_img_per_file, batch_size):
            gal_idx = gal_shuffle[cbatch:(cbatch + batch_size)]
            data_bundle = get_batch_from_fits(filename, noiseless_img_hdu=noiseless_img_hdu,
                                               targets_hdu=targets_hdu, psf_hdu=psf_hdu,
                                               idx_list=gal_idx,
                                               noise_std=noise_std, SNR=SNR,
                                               image_dim=image_dim,
                                               image_per_row=image_per_row,
                                               deconv_mode=deconv_mode,
                                               rho_fista=rho_fista,
                                               risktype=risktype,reg=reg,reg_frac=reg_frac,
                                               tol=tol, shape_constraint=shape_constraint, 
                                               win_filename=winname,win_hdu=win_hdu, mom_hdu=mom_hdu)
            X_train = data_bundle[0]
            y_train = data_bundle[1]
            W_train = data_bundle[2]
            yield (X_train, y_train, W_train)

def npy_batches(files, batch_size=32, nb_img_per_file=10000):
    """Generator function used by the training function fit_generator in Keras.
    It is a function with an infinite loop generating one batch at a time (yield)from npy files.
    The fit_generator process will call this function by itself to get only the number of batches needed without overloading the memory.
    
    :param files: path to the npy files
    :param batch_size: number of images in one batch.
    :param nb_img_per_file: number of images in npy file.
    :type files: list of string
    :type batch_size: int
    :type nb_img_per_file: int

    :returns: yields list of processed images, targets
    :rtype: list of 2D npy.ndarray,list of 2D npy.ndarray
    
    .. note::
        Contrary to :py:func:`dynamic_batches` the npy files have been already preprocessed here (already deconvolved).
        Arbitrary noise addition for each batch is therefore not enforced.
        The fit_generator process will call this function by itself to get o
 
    """
    counter = 0
    while True:
        if counter == 0:
            random.shuffle(files)
        fname = files[counter]
        counter = (counter + 1) % len(files)
        gal_shuffle = np.random.permutation(nb_img_per_file)
        for cbatch in range(0, nb_img_per_file, batch_size):
            gal_idx = gal_shuffle[cbatch:(cbatch + batch_size)]
            data_bundle = np.load(fname)[:,gal_idx,:,:,:]
            X_train = data_bundle[0]
            y_train = data_bundle[1]
            yield (X_train, y_train)

def admm_data(filename, idx, noise_std=None, SNR=None,
                        noiseless_img_hdu=0, targets_hdu=0, psf_hdu=0,
                        image_dim=96, image_per_row=100, deconv_mode=None,
                        risktype="GCV",reg="Dirac",reg_frac=1.0,tol=1e-12):
    """Generator function used by the training function fit_generator in Keras.
    It is a function with an infinite loop generating one batch at a time (yield).
    The fit_generator process will call this function by itself to get only the number of batches needed without overloading the memory.
    
    :param filename: path to the FITS file
    :param idx: index of image to extract
    :param noise_std: white noise standard deviation or range of standard deviation to add (superseded by SNR)
    :param SNR: desired output SNR (measured as :math:`SNR=\\frac{\\|X\\|_2}{\\sigma}`)
    :param noiseless_img_hdu: hdu corresponding to the noiseless image
    :param targets_hdu: hdu corresponding to the target image
    :param psf_hdu: hdu corresponding to the psf image
    :param image_dim: Size of one side of one square image composing the mosaic.
    :param image_per_row: Number of images in one row of the mosaic (assumed square).
    :type filename: string
    :type idx: int
    :type noise_std: float or [float,float]
    :type SNR: float or [float,float]
    :type noiseless_img_hdu: int
    :type targets_hdu: int
    :type psf_hdu: int
    :type image_dim: int
    :type image_per_row: int

    :returns: processed image, target, psf, SNR, noise_std
    :rtype: 2D npy.ndarray,2D npy.ndarray,2D npy.ndarray,float,float
    
    """

    noiseless_img, targets, psfs = format_dataset(filename,
                   noiseless_img_hdu=noiseless_img_hdu,
                   targets_hdu=targets_hdu, psf_hdu=psf_hdu,
                   idx_list=[idx],
                   image_dim=image_dim,
                   image_per_row=image_per_row)
    
    noisy_img, SNR_list, sigma_list = add_noise(noiseless_img, std=noise_std, SNR=SNR)
    if deconv_mode == 'TIKHONOV_HYP':
        #Start by estimating noise
        var_est=mad_est_white_noise_list(noisy_img,wavelet_name='db3')**2
        if(reg=="Lap"):
            trans_reg,_=tf_lap(2,noisy_img[0].shape,is_real=False)
        else:
            trans_reg,_=tf_dirac(2,noisy_img[0].shape,is_real=False) 
        #Estimate hyperparameter and deconvolved image
        deconv_img,hyp=tikhonov_hyp(noisy_img,psfs,trans_reg,var_est,risktype=risktype,tol=tol,reg_frac=reg_frac)
        return noisy_img[0], targets[0], psfs[0], SNR_list[0], sigma_list[0],deconv_img[0],hyp[0]

    else:
        return noisy_img[0], targets[0], psfs[0], SNR_list[0], sigma_list[0]
