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
import starlets
import cadmos_lib as cl

# Define function
def starlet_op(signal):
    n_scale = 4
    bool_gen = True
    return starlets.star2d(signal,scale=n_scale,gen2=bool_gen)

def estimate_thresholds(im,psf):
    """This method estimates the standard deviation map of propagated 
    normalised noise in the starlet space.
    INPUT: None
    OUTPUT: std_map, 3D Numpy Array"""
    psf_rot = cl.rotate180(psf)
    sigma = cl.sigma_mad(im) / np.sqrt(np.sum(psf ** 2))
    def noise_op(res):
        """This function backprojects the noise to the image space.
        INPUT: res, 2D Numpy Array
        OUTPUT: bp_res, 2D Numpy Array"""
        res = cl.convolve(res,psf,'same')
        bp_res = np.array(cl.comp_grad(res,psu,mu,gamma))
        bp_res = cl.convolve(bp_res,psf_rot,'same')
        return bp_res
    noise = sigma*np.random.randn(n_maps,n_row,n_col)
    #noise backprojection
    bp_noise = np.array([noise_op(n) for n in noise])
    #Starlet transforms of noise
    starlet_noise = np.array([starlet_op(bn) for bn in bp_noise])
    #estimate the noise standard deviation condering every noise
    #realisation for every pixel in every scale
    sigma_map = np.array([[[np.std(y) for y in pos] for pos in scale] \
                             for scale in np.moveaxis(starlet_noise,0,-1)])
    thresholds = np.vstack(([(k+1)*s for s in sigma_map[:1]],\
                                     [k*s for s in sigma_map[1:]]))
    return thresholds

# Load Data
f = open(data_path+"meerkat_batch.pkl", "rb")
batch = pickle.load(f)
f.close()
    
n_batch,n_row,n_col = batch['psf'].shape

# prepare score parameters and variables 
n_shearlet = 3
gamma = 0.5
k = 4
n_maps = 100

U = cl.makeUi(n_row,n_col)
_,shearlets_adj = cl.get_shearlets(n_row
                                   ,n_col
                                   ,n_shearlet)
#Adjoint shealret transform of U, i.e Psi^{Star}(U)
psu = np.array([cl.convolve_stack(ui,shearlets_adj) for ui in U])
mu = cl.comp_mu(psu)

# set galaxies and psfs
gals_obs = batch['inputs'][2*768:3*768]
psfs = batch['psf'][2*768:3*768]
tikhos = batch['inputs_tikho'][2*768:3*768]

# Initiate instance of score
# set the value of gamma
g05 = score(gamma=gamma,rip=False,verbose=False)


# Run SCORE

#loop
sol_g05 = []
i=1
for obs, psf, tikho in zip(gals_obs,psfs,tikhos):
    #compute thresholds
    thresholds = estimate_thresholds(obs,psf)
    #deconvolve
    g05.deconvolve(obs=obs,psf=psf,thresholds=thresholds,first_guess=tikho)
    sol_g05 += [g05.solution]
    if i%10 == 0:
        print(i)
    i += 1

filename = data_path + 'score_radio_tikho_g05_3'
np.save(filename,np.array(sol_g05))
