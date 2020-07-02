#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 10:17:16 2018

@author: fsureau 

Deconvolution algorithms.
"""

import os, sys
if os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    
import numpy as np
from skimage import restoration
from DeepDeconv.utils.conv_utils import get_conv2d,recenter_psf
from DeepDeconv.utils.data_utils import max_sv


from DeepDeconv.utils.min_risk_utils import min_risk_est_1d,cy_pereyra_hyper

def tikhonov_deconv_l2(X_fft,trans_func,trans_reg,tau):
    """Tikhonov deconvolution for single image.
    
    :param X_fft: 2D UFT of image see restoration.uft.ufft2
    :param trans_func: 2D transfer function of psf see restoration.uft.ir2tf
    :param trans_reg: 2D transfer function of regularization kernel see restoration.uft.ir2tf
    :param tau: Regularisation parameter of quadratic regularizer.
    :type X_fft: 2D npy.ndarray
    :type trans_func: 2D npy.ndarray
    :type trans_reg: 2D npy.ndarray
    :type tau: 2D npy.ndarray

    :returns: deconvolved image using quadratic regularization
    :rtype: 2D npy.ndarray

    .. warning:: :math:`\\tau` must be positive (not checked)
    .. warning:: :math:`\\tau` contains noise variance (:math:`\\tau=\\mu*\\sigma^2` if :math:`\\mu` is the quadratic prior weight)
    """
    
    hfstar=np.conj(trans_func)
    h2=np.abs(trans_func)**2
    d2=np.abs(trans_reg)**2
    filter_f=hfstar/(h2+tau*d2)
    sol=np.real(restoration.uft.uifft2(filter_f*X_fft))
    return sol


def tikhonov_hyp_single(X,psf,trans_reg,sigma2,risktype='SurePred',tol=1e-12,reg_frac=1.0):
    """Tikhonov deconvolution with hyperparameter selection for a single image.

    :param X: 2D image to deconvolve
    :param psf: point spread function associated to image
    :param trans_reg: 2D transfer function of regularization kernel see restoration.uft.ir2tf
    :param sigma2: White noise variance
    :param risktype: risk to minimize
    :param tol: threshold in power spectrum to set orthogonal of null space for SURE projection risk
    :param reg_frac: fraction of selected hyperparameter to take for deconvolution
    :type X: 2D npy.ndarray 
    :type psf: 2D npy.ndarray 
    :type trans_reg: 2D npy.ndarray
    :type sigma2: double
    :type risktype: string among "SureProj","SurePred" or "GCV"
    :type tol: double (default: 1e-12)
    :type reg_frac: double (default: 1)
    :returns: deconvolved image with associated  hyperparameters
    :rtype: 2D npy.ndarray, double 

    .. warning:: :math:`\\tau` must be positive (not checked)
    .. warning:: :math:`\\tau` contains noise variance (:math:`\\tau=\\mu*\\sigma^2` if :math:`\\mu` is the quadratic prior weight)
    """
    if not len(X.shape) == 2:
        raise TypeError("Input must be a 2D image")

    X_fft=restoration.uft.ufft2(X) # FFT of noisy image
    psf_ctr=recenter_psf(psf,np.array([-0.5,-0.5]))#To recenter for UFFT the kernel
    trans_func = restoration.uft.ir2tf(psf_ctr, X.shape, is_real=False)
    psf_ps=np.abs(trans_func)**2 # power spectrum of psf: |h_w|^2 
    reg_ps=np.abs(trans_reg)**2 #power spectrum of regularization kernel |l_w|^2
    if risktype=="SureProj":
        lst_nonz=np.where(psf_ps>=tol) #'orthogonal of null space' selected
        X_ps_nonz=np.abs(X_fft[lst_nonz])**2 #PS of noisy image
        psf_ps_nonz=np.abs(trans_func[lst_nonz])**2 #|h_w|^2 
        reg_ps_nonz=np.abs(trans_reg[lst_nonz])**2 # |l_w|^2 in case of laplacian
        hyp=min_risk_est_1d(psf_ps_nonz,X_ps_nonz,reg_ps_nonz,
                                sigma2,"Bounded",risktype="SureProj",mu0=1.0).x
    else:
        X_ps=np.abs(X_fft)**2 #PS of noisy image
        if risktype=="Pereyra":
            #choose alpha=beta=1, tau0=1
            hyp=cy_pereyra_hyper(1.0,1.0,1.0, psf_ps.flatten(),X_ps.flatten(),
                                 reg_ps.flatten(),X.size,100,sigma2,marg=False)*sigma2/2.0
        else:
            hyp=min_risk_est_1d(psf_ps.flatten(),X_ps.flatten(),reg_ps.flatten(),
                                sigma2,"Bounded",risktype=risktype,mu0=1.0).x
        
    return (tikhonov_deconv_l2(X_fft,trans_func,trans_reg,hyp*reg_frac),hyp)
  
def tikhonov_hyp(X,psf,trans_reg,sigma2,risktype='SurePred',tol=1e-12,reg_frac=1.0):
    """Wrapper around Tikhonov deconvolution :py:func:`tikhonov_hyp_single` for an image or set of images
    
    :param X: image or set of images to deconvolve
    :param psf: point spread function associated to each image
    :param trans_reg: 2D transfer function of regularization kernel see restoration.uft.ir2tf
    :param sigma2: White noise variance
    :param risktype: risk to minimize
    :param tol: threshold in power spectrum to set orthogonal of null space for SURE projection risk
    :param reg_frac: fraction of selected hyperparameter to take for deconvolution
    :type X: 2D npy.ndarray or list of 2D npy.ndarray
    :type psf: 2D npy.ndarray or list of 2D npy.ndarray
    :type trans_reg: 2D npy.ndarray
    :type risktype: string among "SureProj","SurePred" or "GCV"
    :type tol: double (default: 1e-12)
    :type reg_frac: double (default: 1)

    :returns: deconvolved (list of) image(s) with associated (list of) hyperparameters(s)
    :rtype: 2D npy.ndarray or list of 2D npy.ndarray, double or list of doubles

    .. warning:: :math:`\\tau` must be positive (not checked)
    .. warning:: :math:`\\tau` contains noise variance (:math:`\\tau=\\mu*\\sigma^2` if :math:`\\mu` is the quadratic prior weight)
    """
    if len(X.shape) == 2:
        return tikhonov_hyp_single(X,psf,trans_reg,sigma2,risktype=risktype,tol=tol)
    else:
        assert X.shape == psf.shape
        assert sigma2.size == len(psf)
        tikho_dec_list = []
        tikho_hyp_list = []
        for i in range(len(X)):
            deconvolved,hyp_param = tikhonov_hyp_single(X[i],psf[i],trans_reg,
                                                sigma2[i],risktype=risktype,tol=tol,reg_frac=reg_frac)
            tikho_dec_list.append(deconvolved)
            tikho_hyp_list.append(hyp_param)
        return np.asarray(tikho_dec_list),np.asarray(tikho_hyp_list)

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

def FISTA_core(X, psf, rho, max_iter=500):
    """FISTA algorithm with augmented lagrangian for ADMM.
    
    :param X: image (array in 2D) to deconvolve
    :param psf: point spread function, same dimension as X
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
