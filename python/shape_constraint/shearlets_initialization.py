        #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:05:30 2018

@author: fnammour
"""

#%%DATA INITIALIZATION
import numpy as np
from AlphaTransform import AlphaShearletTransform as AST
import cadmos_lib as cl
import os

code_path = '/Path/To/Code/'

row,column = np.array([96,96])
U = cl.makeUi(row,column)

gamma = 1 #trade-off parameter

# Get shearlet elements
#Step 1 : create a shearlet transform instance
trafo = AST(column, row, [0.5]*3,real=True,parseval=True,verbose=False)
#Step 2 : get shearlets filters
shearlets = trafo.shearlets
#Step 3 : get the adjoints
adjoints = cl.get_adjoint_coeff(trafo)

#Normalize shearlets filter banks
#/!\ The order is important/!\
adjoints = cl.shear_norm(adjoints,shearlets)
shearlets = cl.shear_norm(shearlets,shearlets)

#Compute moments constraint normalization coefficients
#the $\Psi^*_j$ are noted adj_U
adj_U = cl.comp_adj(U,adjoints)
mu = cl.comp_mu(adj_U)

#Lipschitz constant of the gradient
#We create a wrapper function to estimate the spectral radius of the S matrix

sigma = 1
cache_path = code_path+'my_cache/{0}x{1}/'.format(row,column)
lip_cst_path = cache_path+'lip_cst_gamma_{}.npy'.format(gamma)
try:
    L = np.load(lip_cst_path)
except FileNotFoundError:
    lip_eps = 1e-3#error upperbound
    
    def grad_Op(R):
        return np.array(cl.comp_grad(R,adj_U,mu,gamma))

    L,_= cl.power_iteration(grad_Op, (row,column),lip_eps)
    del grad_Op
    if not(os.path.exists(cache_path)):
            os.mkdir(cache_path)
    np.save(lip_cst_path,L)