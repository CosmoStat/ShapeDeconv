#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding: utf-8

# Import Libraries
import sys
import pickle

# Define paths and directories
lib_path = '/gpfswork/rech/xdy/uze68md/GitHub/'
data_path = '/gpfswork/rech/xdy/uze68md/data/'

# Add library path to PYTHONPATH
path_alphatransform = lib_path+'alpha-transform'
path_score = lib_path+'score'
sys.path.insert(0, path_alphatransform)
sys.path.insert(0, path_score)

# Load libraries
from score import score
import numpy as np
import matplotlib.pyplot as plt
from cv2 import resize, INTER_AREA

# Define function
def downsample_im(input_im, output_dim):
    """Downsample image.
    Based on opencv function resize.
    [doc](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#void%20resize(InputArray%20src,%20OutputArray%20dst,%20Size%20dsize,%20double%20fx,%20double%20fy,%20int%20interpolation))
    The input image is downsampled to the dimensions specified in `output_dim`.
    The downsampling method is based on the `INTER_AREA` method.
    See [tensorflow_doc](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/resize-area)
    Each output pixel is computed by first transforming the pixel's footprint
    into the input tensor and then averaging the pixels that intersect the
    footprint. An input pixel's contribution to the average is weighted by the
    fraction of its area that intersects the footprint.
    This is the same as OpenCV's INTER_AREA.
    An explanation of the INTER_AREA method can be found in the next
    [link](https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3).
    This version should be consistent with the tensorflow one.
    Parameters
    ----------
    input_im: np.ndarray (dim_x, dim_y)
        input image
    output_dim: int
        Contains the dimension of the square output image.
    """
    return resize(input_im, (output_dim, output_dim), interpolation=INTER_AREA)

# Load Data
f = open(data_path+"cfht_batch.pkl", "rb")
batch = pickle.load(f)
f.close()
    
n_batch = batch['psf_hst'].shape[0]
# generate the psfs in the spatial domain
shape = batch['inputs'].shape[-1]
psf_tile_partial = np.array([c/h for c,h in zip(batch['psf_cfht'],batch['psf_hst'])])
psf_tile_partial = np.array([downsample_im(np.fft.ifftshift(np.fft.irfft2(p)),shape) for p in psf_tile_partial])
psf_tile_partial = np.array([p/p.sum() for p in psf_tile_partial])

gals_obs = batch['inputs']
psfs = psf_tile_partial

# Initiate instance of score
# set the value of gamma
g0 = score(gamma=0,rip=False,verbose=False)


# Run SCORE

#loop
sol_g0 = []
i=1
for obs, psf in zip(gals_obs,psfs):
    #deconvolve
    g0.deconvolve(obs=obs,psf=psf)
    sol_g0 += [g0.solution]
    if i%10 == 0:
        print(i)
    i += 1

filename = data_path + 'score_g0'
np.save(filename,np.array(sol_g0))
