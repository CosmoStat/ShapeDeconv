#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 10:16:40 2018

@author: alechat


Contains all the functions needed for pre-processing stuff.
"""

import numpy as np
from numpy.fft import fft2
import pywt
import scipy.stats

def add_noise(X, std=None, SNR=None):
    """Add noise to an image of array of images.
    
    :param X: noiseless image or set of images.
    :param std: standard deviation of the noise or bounds
    :param SNR: signal to noise ratio or bounds
    :type X: 2D or 3D npy.ndarray 
    :type std: float or [float,float]
    :type SNR: float or [float,float]

    :returns: noisy image(s),SNR value(s),std value(s).
    :rtype: 2D or 3D npy.ndarray, npy.ndarray, npy.ndarray

    .. note::
        X can be an image or a set of images.
        Only input the std or the SNR. Giving a list of 2 values as std or SNR will define the bounds
        and the noise will be added with a random value picked between those bounds.
    """
    Xnoise = X.copy()
    if len(Xnoise.shape)==2:
        Xnoise = Xnoise.reshape(1,Xnoise.shape[0],Xnoise.shape[1])
    SNR_list = []
    sigma_list = []
    if SNR is None and std is not None:
        if type(std)==list:
            min_std = np.min(std)
            max_std = np.max(std)
        else:
            noise_std = std
        np.random.seed(1)
        for i in range(len(Xnoise)):
            normX = np.linalg.norm(Xnoise[i])
            if type(std)==list:
                noise_std = np.random.random()*(max_std-min_std)+min_std
            noise = np.random.normal(0, noise_std, size=Xnoise[i].shape)
            SNR_list.append(normX/noise_std)
            sigma_list.append(noise_std)
            Xnoise[i] += noise
    elif SNR is not None:
        for i in range(len(Xnoise)):
            normX = np.linalg.norm(Xnoise[i])
            if type(SNR)==list:
                max_std = normX/np.min(SNR)
                min_std = normX/np.max(SNR)
                noise_std = np.random.random()*(max_std-min_std)+min_std
            else:
                noise_std = normX/SNR
            noise = np.random.normal(0, noise_std, size=Xnoise[i].shape)
            SNR_list.append(normX/noise_std)
            sigma_list.append(noise_std)
            Xnoise[i] += noise
    else:
        raise ValueError('Incorrect value for noise std or SNR')
    return Xnoise.reshape(X.shape), np.asarray(SNR_list), np.asarray(sigma_list)

def mad_est_white_noise_single(noisy_im,wavelet_name='db3'):
    """MAD estimator of noise standard deviation in wavelet space
    
    :param noisy_im: 2D noisy image 
    :param wavelet_name: choice of type of pywt wavelet
    :type noisy_im: 2D npy.ndarray 
    :type wavelet_name: string describing pywt wavelet :seealso:pywt.families

    :returns: estimated standard deviation of the noise
    :rtype: double

    .. note:: No coefficient affected by boundary effects is used
    """
    wavelet = pywt.Wavelet(wavelet_name)
    shift=wavelet.dec_len//2
    _,det = pywt.dwt2(noisy_im, wavelet_name)
    cst=1.0/scipy.stats.norm.ppf(0.75, loc=0, scale=1)
    med=np.median(det[2][shift:-shift,shift:-shift])
    return cst*np.median(np.abs(det[2][shift:-shift,shift:-shift]-med))


def mad_est_white_noise_list(set_im,wavelet_name='db3'):
    """Wrapper around MAD estimator of noise standard deviation in wavelet space for lists
    
    :param noisy_im: list of 2D noisy images
    :param wavelet_name: choice of type of pywt wavelet
    :type noisy_im: list of 2D npy.ndarray 
    :type wavelet_name: string describing pywt wavelet :seealso:pywt.families

    :returns: list of estimated standard deviation of the noise
    :rtype: list of doubles

    .. note:: No coefficient affected by boundary effects is used
    """
    return np.array([mad_est_white_noise_single(im,wavelet_name=wavelet_name) for im in set_im])


def compute_SNR(X, noise_std):
    """Compute SNR given a set of noiseless images and the std of the noise.
    
    :param X: array of noiseless images.
    :param noise_std: standard deviation of the noise
    :type X: 3D npy.ndarray 
    :type noise_std: float

    :returns: array of all the SNR values.
    :rtype: npy.ndarray
    """
 
    SNR = np.array([np.linalg.norm(d) for d in X])/noise_std
    return SNR

def compute_SNR_approx(X, noise_std):
    """Compute (raw) approximation of SNR given a set of noisy images and the std of the noise.
    
    :param X: array of noisy images.
    :param noise_std: standard deviation of the noise
    :type X: 3D npy.ndarray 
    :type noise_std: float

    :returns: array of all the SNR values.
    :rtype: npy.ndarray
    """

    pixels = X.shape[1] * X.shape[2]
    SNR = np.sqrt(np.abs(np.array([np.linalg.norm(d)**2 for d in X]) - pixels*(noise_std**2)))/noise_std
    return SNR


def max_sv(psf):
    """Compute square of spectral radius of convolution matrix, given convolution kernel.
    
    :param psf: convolution kernel
    :type psf: 2D npy.ndarray 

    :returns: square of spectral radius.
    :rtype: float
    """

    H = fft2(psf)
    normH = np.abs(H.conj() * H)
    return np.max(normH)
