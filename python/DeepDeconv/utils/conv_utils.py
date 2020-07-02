#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 10:17:37 2018

@author: fsureau


Function used to deal with the shift introduced by the convolution when using images with even dimensions.
"""

import numpy as np
import scipy
import copy
from skimage import restoration

def get_conv2d_fftconv(im, psf, mode="scipy_fft", transp=False,extended=True):
    """Apply the convolution between an image and one PSF taking into account the fft convention for the center of images.
    
    :param im: image to convolve
    :param psf: point spread function, centered at (sz_psf[0]+1)//2,(sz_psf[0]+1)//2
    :param mode: convolution mode, either 'scipy_fft' or 'scipy' to do Fourier space or direct space convolution.
    :param transp: convolve with transpose matrix (kernel with FT the conjugate of the kernel) or with original convolution matrix
    :type im: 2D npy.ndarray 
    :type psf: 2D npy.ndarray 
    :type mode: string
    :type transp: bool

    :returns: convolved image
    :rtype: 2D npy.ndarray
    
    .. note::
        Here we compute the kernel with FT the conjugate of the FFT of the kernel to get the transpose matrix.
    """


    #THIS ROUTINE ASSUMES THAT THE PSF CTR IS AT (sz_psf[0])//2,(sz_psf[0])//2, which is the center chosen

    psf_ctr=(np.array(psf.shape))//2
    sh_im_in=im.shape
    if transp:
        ext_size=np.array(psf.shape)+np.array(im.shape)-1
        ext_psf_ctr=np.zeros(ext_size)
        roll_im=ext_size//2-psf_ctr #This is the shift to center the extended psf if the psf is originally centered 
        sl=slice(0,psf.shape[0]),slice(0,psf.shape[1])
        ext_psf_ctr[sl]=psf
        ext_psf_ctr=np.roll(ext_psf_ctr,-psf_ctr,axis=(0,1))
        ext_psf=np.real(restoration.uft.uifft2(np.conj(restoration.uft.ufft2(ext_psf_ctr))))
        ext_psf=np.roll(ext_psf,psf_ctr,axis=(0,1))[sl]        
        #ext_psf_ctr=np.zeros(np.array(psf.shape)*2)
        #sl=(slice(psf_ctr[0],psf_ctr[0]+psf.shape[0]),slice(psf_ctr[1],psf_ctr[1]+psf.shape[1]))
        #ext_psf_ctr[sl]=psf
        #ext_psf = (np.real(restoration.uft.uifft2(np.conj(restoration.uft.ufft2(ext_psf_ctr)))))[sl]
    else:
        ext_psf=psf
    if(mode=="scipy_fft"):
        conv=scipy.signal.fftconvolve(im, ext_psf,mode="full")[psf_ctr[0]:psf_ctr[0]+sh_im_in[0],psf_ctr[1]:psf_ctr[1]+sh_im_in[1]] 
    elif (mode=="scipy"):
        conv=scipy.signal.convolve2d(im, ext_psf,mode="full")[psf_ctr[0]:psf_ctr[0]+sh_im_in[0],psf_ctr[1]:psf_ctr[1]+sh_im_in[1]] 
    return conv


def get_conv2d(im, psf, mode="scipy_fft", transp=False):
    """Apply the convolution between an image and one PSF taking into account the astropy/scipy convention for the center of images.
    
    :param im: image to convolve
    :param psf: point spread function
    :param mode: convolution mode, either 'scipy_fft' or 'scipy' to do Fourier space or direct space convolution.
    :param transp: convolve with transpose matrix (kernel rotated by 180 degrees) or with original convolution matrix
    :type im: 2D npy.ndarray 
    :type psf: 2D npy.ndarray 
    :type mode: string
    :type transp: bool

    :returns: convolved image
    :rtype: 2D npy.ndarray
    
    .. note::
        The image is first extended so the dimensions are odd in order to have a defined center. 
        The convolution is then done before cropping the image back.

    """


    #THIS ROUTINE ASSUMES THAT THE PSF CTR IS AT (sz_psf[0]-1)//2,(sz_psf[0]-1)//2, wich is the center chosen by astropy
    sh_im_in=im.shape
    sh_psf_in=psf.shape
    sh_psf_out=np.array(sh_psf_in)
    start_transp = sh_psf_out*0
    #ENSURE ALWAYS ODD SIZE FOR PSF
    cur_slice = []
    for ks,vsh in enumerate(sh_psf_out):
        if(vsh%2 ==0):
            sh_psf_out[ks]+=1
            start_transp[ks]=-2 # TWICE BECAUSE 1) WE EXTEND 2) THE CENTER IS SHIFTED BY 1
        else:
            start_transp[ks]=0 # 0 BECAUSE 1) WE DO NOT EXTEND 2) THE CENTER IS AT THE CENTER
        cur_slice.append(slice(0, sh_psf_in[ks],1))
    ext_psf =np.zeros(sh_psf_out)
    ext_psf[cur_slice]= psf
    
    psf_ctr=(np.array(psf.shape)-1)//2
    if transp:
        ext_psf=np.rot90(ext_psf,2)
        psf_ctr=psf_ctr-start_transp
    if(mode=="scipy_fft"):
        conv=scipy.signal.fftconvolve(im, ext_psf,mode="full")[psf_ctr[0]:psf_ctr[0]+sh_im_in[0],psf_ctr[1]:psf_ctr[1]+sh_im_in[1]] 
    elif (mode=="scipy"):
        conv=scipy.signal.convolve2d(im, ext_psf,mode="full")[psf_ctr[0]:psf_ctr[0]+sh_im_in[0],psf_ctr[1]:psf_ctr[1]+sh_im_in[1]] 
    return conv
    
def perform_shift_in_frequency(fpsf, size_img, shift):
    """Add linear phase to fourier transform to shift signal centered in *shift* to 0
    
    :param fpsf: fourier transform needing extra phase factor
    :param size_img: size of input image in [x,y] (to check if real or complex transform)
    :param shift: xshift in [x,y] for array[x,y]
    :type fpsf: 2D complex npy.ndarray 
    :type size_img: list of 2 floats
    :type shift: list of 2 floats

    :returns: fourier transform with extra phase (same size as fpsf)
    :rtype: 2D complex npy.ndarray 
    """
 
    phase_factor= np.float64(2. * np.pi) * shift.astype(np.float64)
    if phase_factor[0] ==0.:
        kx_ft=np.zeros(size_img[0])+1.
    else :
        kx_ft=np.exp(np.fft.fftfreq(size_img[0],d=1./phase_factor[0])*1j)
    if phase_factor[1] ==0.:
        ky_ft=np.zeros(fpsf.shape[1],dtype=np.float64)+1.
    else:
        if fpsf.shape[1] != size_img[1]:
            ky_ft=np.exp(np.fft.rfftfreq(size_img[1],d=1./phase_factor[1])*1j)
        else:
            ky_ft=np.exp(np.fft.fftfreq(size_img[1],d=1./phase_factor[1])*1j)
    return copy.deepcopy(np.outer(kx_ft,ky_ft)*fpsf)

def correct_pixel_window_function(fpsf, size_img):
    """Correct for pixel window effect (beware of aliasing)
    
    :param fpsf: fourier transform needing extra phase factor
    :param size_img: size of input image in [x,y] (to check if real or complex transform)
    :type fpsf: 2D complex npy.ndarray 
    :type size_img: list of 2 floats

    :returns: the fourier transform with pixel window function correction (same size as fpsf)
    :rtype: 2D complex npy.ndarray 
    
    .. note::
        This is useful for convolution with band limited signal sampled higher than Nyquist frequency,
        to better approximate continuous convolution followed by sampling with discrete convolution.
    """

    mult_x=np.array(np.fft.fftfreq(size_img[0]),dtype=np.float64)
    if fpsf.shape[1] != size_img[1]:
       mult_y=np.array(np.fft.rfftfreq(size_img[1]),dtype=np.float64)
    else:
       mult_y=np.array(np.fft.fftfreq(size_img[1]),dtype=np.float64)
    pwf_x=np.array([np.sinc(kx) for kx in mult_x],dtype=np.float64)
    pwf_y=np.array([np.sinc(ky) for ky in mult_y],dtype=np.float64)
    return copy.deepcopy(fpsf / np.outer(pwf_x, pwf_y))
    
    
def recenter_psf(psf,shift):
    """Recenter PSF by adding linear phase in Fourier domain see :py:func:`perform_shift_in_frequency`.
    
    :param psf: point spread function to recenter (e.g. center at the intersection of 4 pixels)
    :param shift: shift in [x,y] for array[x,y] to recenter; e.g. centering implies [-0.5,-0.5] in great3
    :type psf: 2D npy.ndarray 
    :type shift: list of 2 floats

    :returns: the recentered psf
    :rtype: 2D npy.ndarray 
    
    .. note::
        This is useful when the center of an image and psf is at the intersection of 4 pixels. In this case,
        one needs to recenter the psf so that the convolved result is also centered at the intersection of 4 pixels.
    """
    fpsf=np.fft.fft2(psf)
    fpsf_ctr=perform_shift_in_frequency(fpsf, psf.shape, shift)
    return np.real(np.fft.ifft2(fpsf_ctr))
    
def tf_dirac(ndim,shape,is_real=False):
    """Transfer function of the Dirac kernel.
    
    :param ndim: number of dimensions of kernel
    :param shape: size of input image
    :param is_real: whether to use hermitian symmetry of FFT and save space in fourier domain
    :type ndim: int
    :type shape: list of ndim floats
    :param is_real: bool

    :returns: the transfer function corresponding to dirac (= all 1 in fourier domain),dirac kernel
    :rtype: 2D complex npy.ndarray, 2D npy.ndarray
    
    .. note::
        This transfer function can be directly multiplied with the FFT of an image to obtain a convolved image.
    """
    
    impr = np.zeros([3] * ndim)
    impr[(slice(1, 2), ) * ndim] = 1.0 
    return restoration.uft.ir2tf(impr, shape, is_real=is_real), impr

def tf_lap(ndim,shape,is_real=False):
    """Transfer function of the Laplacian kernel.
    
    :param ndim: number of dimensions of kernel
    :param shape: size of input image
    :param is_real: whether to use hermitian symmetry of FFT and save space in fourier domain
    :type ndim: int
    :type shape: list of ndim floats
    :param is_real: bool

    :returns: the transfer function corresponding to Laplacian kernel, Laplacian kernel
    :rtype: 2D complex npy.ndarray, 2D npy.ndarray
    
    .. note::
        This transfer function can be directly multiplied with the FFT of an image to obtain a convolved image.
    """
   
    return restoration.uft.laplacian(ndim, shape, is_real=is_real)

