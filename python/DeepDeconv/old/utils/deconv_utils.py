#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 10:17:16 2018

@author: alechat


All the deconvolution algorithms used.
"""

import os, sys
if os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    
import numpy as np
from skimage import restoration
from DeepDeconv.utils.conv_utils import get_conv2d
from DeepDeconv.utils.data_utils import max_sv


def tikhonov(X, psf, tau):
    """Tikhonov deconvolution using Laplacian kernel regularization for image or set of images using skimage.
    
    :param X: image or set of images to deconvolve
    :param psf: psf image or set of images, same dimension as X
    :param tau: Regularisation parameter of Laplacian kernel quadratic regularizer or list of regularisation parameters.
    :type X: 2D or 3D npy.ndarray
    :type psf: 2D or 3D npy.ndarray
    :type tau: float or list of floats

    :returns: deconvolved image using laplacian kernel quadratic regularization
    :rtype: 2D or 3D npy.ndarray

    .. warning:: :math:`\\tau` must be positive (not checked)
    .. warning:: :math:`\\tau` contains noise variance (:math:`\\tau=\\mu*\\sigma^2` if :math:`\\mu` is the quadratic prior weight)
    """

    if len(X.shape) == 2:
        return(restoration.wiener(X, psf, tau))
    tikho_list = []
    for i in range(len(X)):
        deconvolved = restoration.wiener(X[i], psf[i], tau[i])
        tikho_list.append(deconvolved)
    return np.asarray(tikho_list)

def FISTA_core(X, psf, rho, sigma=None, max_iter=500):
    """FISTA algorithm with augmented lagrangian for ADMM.
    
    :param X: image to deconvolve
    :param psf: psf image, same dimension as X
    :param rho: augmented lagrangian parameter.
    :param sigma: white noise standard deviation.
    :param max_iter: maximal number of iterations.
    :type X: 2D npy.ndarray
    :type psf: 2D npy.ndarray
    :type rho: float 
    :type sigma: float 
    :type max_iter: long int 

    :returns: deconvolved image 
    :rtype: 2D npy.ndarray

    .. warning:: :math:`\\rho` must be positive (not checked)
    .. warning:: :math:`\\sigma` must be positive (not checked)
    """
 
    hth = get_conv2d(psf, psf, mode='scipy_fft', transp=True)
    hty = get_conv2d(X, psf, mode='scipy_fft', transp=True)
    x_fista_old = np.zeros(X.shape)
    z_fista_old = np.zeros(X.shape)
    t_fista_old = 1
    alpha = 1. / ((max_sv(psf)+rho)*(1+1e-5))
    for i in range(max_iter):
        grad = get_conv2d(z_fista_old, hth, mode='scipy_fft', transp=False) - hty
        x_fista_new = z_fista_old - alpha * (grad + rho * z_fista_old)
        x_fista_new[x_fista_new<0] = 0
        t_fista_new = (1. + np.sqrt(4. * t_fista_old**2 + 1.))/2.
        lambda_fista = 1 + (t_fista_old - 1)/t_fista_new
        z_fista_new = x_fista_old + lambda_fista * (x_fista_new - x_fista_old)
        np.copyto(x_fista_old, x_fista_new)
        np.copyto(z_fista_old, z_fista_new)
        t_fista_old = t_fista_new
    return x_fista_old

    
def FISTA(obs, psf, rho, max_iter=500):
    """Wrapper around FISTA :py:func:`FISTA_core` for a set of images
    
    :param obs: 3D set of images to deconvolve.
    :param psf: associated 3D set of point spread functions
    :param rho: hyperparameter of the augmented lagrangian.
    :param max_iter: maximal number of iterations
    :type obs: 3D npy.ndarray 
    :type psf: 3D npy.ndarray
    :type rho: float
    :type max_iter: int

    :returns: set of deconvolved images
    :rtype: 3D npy.ndarray

    .. warning:: :math:`\\rho` must be positive (not checked)
    """ 
    
    result = np.zeros(obs.shape)
    for gal_idx in range(len(obs)):
        result[gal_idx,:,:] = FISTA_core(obs[gal_idx], psf[gal_idx], rho, max_iter=max_iter)
    return result
