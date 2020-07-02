#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 10:27:22 2018

@author: alechat

Read fits files and write them.
"""

import os
import numpy as np
from astropy.io import fits as fits

def stampCollection2Mosaic(stamplist,image_dim=96,image_per_row=100):
    """Construct a mosaic of stamps from a list of stamps.
    
    :param stamplist: list of 2D stamps
    :param image_dim: Size of one side of one square image composing the mosaic.
    :param image_per_row: Number of images in one row/column of the mosaic (which is supposed square).
    :type stamplist: list of 2D npy.ndarray
    :type image_dim: int
    :type image_per_row: int

    :returns: mosaic of all stamps
    :rtype: 2D npy.ndarray

    """

    mosaic=np.empty((image_per_row*image_dim,image_per_row*image_dim))
    nb_gal=image_per_row*image_per_row
    for i in range(nb_gal):
        y = (image_dim*i)%(image_per_row*image_dim)
        x = i//image_per_row * image_dim
        mosaic[x:x+image_dim,y:y+image_dim]=stamplist[i,:,:]
    return mosaic   



def fits2npy(filename, idx_list, hdu, image_dim=96, image_per_row=100):
    """From a given HDU inside a FITS file, process the mosaic of images into a list of 2D patches.
    
    :param filename: pathname to the FITS file.
    :param idx_list: indices of the images to retrieve inside the HDU.
    :param hdu: hdu to retrieve in FITS file
    :param image_dim: Size of one side of one square image composing the mosaic.
    :param image_per_row: Number of images in one row/column of the mosaic (which is supposed square).
    :type filename: string
    :type idx_list: list of int
    :type hdu: int
    :type image_dim: int
    :type image_per_row: int

    :returns: list of all patches
    :rtype: list of 2D npy.ndarray

    """

    data = fits.getdata(filename, hdu)
    data_array = []
    for i in idx_list:
        y = (image_dim*i)%(image_per_row*image_dim)
        x = i//image_per_row * image_dim
        data_array.append(data[x:x+image_dim,y:y+image_dim])
    return np.asarray(data_array)

def save2fits(data, filename, image_dim=96, image_per_row=100, image_per_col=100):
    """Save a list of numpy array containing a set of patches arranged as a mosaic into a fits file.
    
    :param data: array of all images
    :param filename: pathname to the FITS file.
    :param image_dim: Size of one side of one square image composing the mosaic.
    :param image_per_row: Number of images in one row of the mosaic.
    :param image_per_col: Number of images in one column of the mosaic.
    :type data: npy,ndarray
    :type filename: string
    :type image_dim: int
    :type image_per_row: int
    :type image_per_col: int

    :returns:mosaic of all the images
    :rtype: 2D npy.ndarray

    """
    
    ## Convert the array into one image containing the mosaic.
    nb_img = len(data)
    fits_img = np.zeros((image_dim*image_per_row, image_dim*image_per_col))
    for i in range(nb_img):
        y = (image_dim*i)%(image_per_col*image_dim)
        x = i//image_per_row * image_dim
        fits_img[x:x+image_dim,y:y+image_dim] = data[i].reshape((image_dim,image_dim))
        
    ## Write the file (Delete existing file with the same name beforehand)
    try:
        os.remove(filename)
    except OSError:
        pass
    fits.writeto(filename, fits_img)
    return fits_img

