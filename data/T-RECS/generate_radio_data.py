#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:57:00 2019

@author: fnammour
"""

import numpy as np
from modopt.signal.wavelet import filter_convolve
import os

bool_extracted = True
is_saved = True

def fft(img):
    img_fft = np.fft.fft2(img)
    return np.fft.fftshift(img_fft)

def ifft(img):
    img_shifted = np.fft.ifftshift(img)
    return np.real(np.fft.ifft2(img_shifted))

def gen_noise(gal_num,n_row,n_column, PSF):
    noise_i = np.random.rand(gal_num, n_row, n_col)
#    noise_fft = np.array([fft(n) for n in noise_i])
#    noise_psf_fft = np.array([n*PSF_fft for n in noise_fft])
#    noise_m = ifft(noise_psf_fft)
    #REPLACING FFT WITH CONVOLUTION
    noise_m = filter_convolve(PSF,noise_i)
    noise_m_std = np.std(noise_m)
    noise = noise_m/noise_m_std
    return noise

def compute_background_mask(img,p=1,q=4,center=None):
    n_lines,n_columns = img.shape
    x_slice,y_slice = p*n_lines//q,p*n_columns//q
    if center == None:
        x_c,y_c = n_lines//2,n_columns//2
    else:
        x_c,y_c=center
    background_mask = np.ones(img.shape,dtype=bool)
    background_mask[x_c-x_slice:x_c+x_slice,y_c-y_slice:y_c+y_slice] = False
    return background_mask
    

def compute_SNR(img):
    peak_flux = np.max(np.abs(img))
    n_lines,n_columns = img.shape
    x_max,y_max = np.argmax(img)//96,np.argmax(img)%96
    background = compute_background_mask(img,p=1,q=3,center=(x_max,y_max))
    sigma = np.std(img[background])
    return peak_flux/sigma

def gen_obs(img,noise,SNR):
    peak_flux = np.max(img)
    return (img + noise * peak_flux / SNR)

#LOAD DATA
gals_path = '/Users/fnammour/Documents/Thesis/direct_deconvolution/Data/T_RECS/'
gals_cat_path = '/Users/fnammour/Documents/Thesis/direct_deconvolution/T-RECS_simulations/'

PSF_num = 5

if bool_extracted:
    gals = np.load(gals_path+'galaxies.npy')
else:
    gal_cat = np.load(gals_cat_path+'Cat-SFG.npz')
    gal_items = gal_cat.items()
    list_gal = gal_items[1]
    gals = list_gal[1]
    if not os.path.exists(gals_path):
        os.makedirs(gals_path)
    np.save(gals_path+'galaxies.npy',gals)
    
gal_num, n_row, n_col = gals.shape

PSF_path = gals_path+'PSF_{0}asec/'.format(PSF_num)

PSF = np.load(PSF_path+'PSF_{0}asec.npy'.format(PSF_num))

##COMPUTE FOURIER TRANSFORM
#gals_fft = np.array([fft(gal) for gal in gals])
#PSF_fft = fft(PSF)
#
##GENERATE DATA
##compute visibilities
#visib = np.array([gal*PSF_fft for gal in gals_fft])
#gals_visib = np.array([ifft(v) for v in visib])
#REPLACING FFT WITH CONVOLUTION
gals_visib = filter_convolve(PSF,gals)
np.save(PSF_path+'convolved_galaxies.npy',gals_visib)

#Generate observation
SNRs = [10]#[5,10,20,50,100]

for SNR in SNRs:
    #compute noise
    #generate complex white additive Gaussian noise
    noise = gen_noise(gal_num,n_row,n_col, PSF)
    gals_obs = np.array([gen_obs(gal,n,SNR) for gal,n in zip(gals_visib,noise)])
    obs_path = PSF_path+'SNR{}/'.format(SNR)
    if not is_saved:
        if not os.path.exists(obs_path):
            os.makedirs(obs_path)
        np.save(obs_path+'noisy_galaxies_SNR{0}_PSF_{1}asec'.format(SNR,PSF_num),gals_obs)
    
    



