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


# Load Data
f = open(data_path+"cfht_batch.pkl", "rb")
batch = pickle.load(f)
f.close()
    
n_batch = batch['psf_hst'].shape[0]
# generate the psfs in the spatial domain
shape = batch['inputs'].shape[1:]
psf_tile_cfht = np.array([np.fft.ifftshift(np.fft.irfft2(p, s=shape)) for p in batch['psf_cfht']])

gals_obs = batch['inputs']
psfs = psf_tile_cfht

# Initiate instance of score
# set the value of gamma
g0 = score(gamma=0,rip=True,verbose=False)


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
