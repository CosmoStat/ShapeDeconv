#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 11:51:32 2018

@author: alechat
"""
import os, sys
if os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import numpy as np
import matplotlib.pylab as plt

from modopt.opt.algorithms import SetUp
from modopt.opt.cost import costObj

from DeepDeconv.utils.conv_utils import get_conv2d
from DeepDeconv.utils.data_utils import max_sv

class Cost_Lagrangian(object):
    def cost(self, *args):
        x, z, mu, rho, obs, psf = args[0]
        cost_val = 0.5 * rho * np.linalg.norm(z-x) ** 2 - np.dot(mu.flatten(), (z-x).flatten()) + 0.5 * np.linalg.norm(obs - get_conv2d(x, psf)) ** 2
        return cost_val

class ADMM(SetUp):
    def __init__(self, x, z, mu, obs, psf, sigma, dnn, mask=np.ones((96,96)),
             rho=50, gamma=1.4, eta=0.5, rho_cap=200,
             convolve_mode='scipy_fft'):
        SetUp.__init__(self)
        
        # Set the initial variable values
        (self._check_input_data(data) for data in (x, z, mu))
        self._x_old = np.copy(x)
        self._z_old = np.copy(z)
        self._mu_old = np.copy(mu)
        self._mask = np.copy(mask)
        
        # Set the algorithm operators
        self._obs = obs
        self._psf = psf
        self._cost_func = costObj([Cost_Lagrangian()], tolerance=1e-10, verbose=False)
        self._delta_old = 0
        
        # Set the algorithm parameters
        (self._check_param(param) for param in (rho, gamma))
        self._rho= rho
        self._gamma = gamma
        self._eta = eta
        self._sigma = sigma
        self._rho_cap = rho_cap
        self._sv = max_sv(psf)
        self._alpha = 1. / ((self._sv+self._rho)*(1+1e-5))
        
        self._convolve_mode = convolve_mode
        
        # Set the Neural Network
        self._DNN = dnn
        
    def _fista(self, max_iter=500):
        x_fista_old = np.copy(self._x_old)
        z_fista_old = np.copy(self._z_old)
        t_fista_old = 1
        for i in range(max_iter):
            grad = get_conv2d(self._mask*get_conv2d(z_fista_old, self._psf, mode='scipy_fft', transp=False) - self._obs, self._psf, mode='scipy_fft', transp=True)
            x_fista_new = z_fista_old - self._alpha * (grad + self._mu_old * self._sigma**2 + self._rho * self._sigma**2 * (z_fista_old - self._z_old))
            x_fista_new[x_fista_new<0] = 0
            t_fista_new = (1. + np.sqrt(4. * t_fista_old**2 + 1.))/2.
            lambda_fista = 1 + (t_fista_old - 1)/t_fista_new
            z_fista_new = x_fista_old + lambda_fista * (x_fista_new - x_fista_old)
            np.copyto(x_fista_old, x_fista_new)
            np.copyto(z_fista_old, z_fista_new)
            t_fista_old = t_fista_new
        return x_fista_old
            
    def _dnn_process(self, input_dnn):
        tmp_inp = input_dnn[47:47+96,47:47+96]
        out_dnn = self._DNN.model.predict(tmp_inp.reshape((1,96,96,1)), batch_size=1, verbose=0)
        outext = np.zeros((193,193))
        outext[47:47+96,47:47+96] = out_dnn.reshape(96,96)
        return outext
        
    def _update(self, verbose=True):
        #X
        self._x_new = self._fista()
        
        #Z
        self._z_tmp = self._x_new + self._mu_old / self._rho
        self._z_new = self._dnn_process(self._z_tmp)
             
        #MU
        self._mu_new = self._mu_old + (self._x_new - self._z_new) * self._rho
        
        #COST
        if verbose:
            if self._cost_func:
                print('COST: %.5f'%self._cost_func._calc_cost([self._x_new, self._z_new, self._mu_new, self._rho, self._obs, self._psf]))
            print('DIST X-Z: %.5f'%np.linalg.norm(self._x_new-self._z_new))
            print('rho = %.5f'%self._rho)
            
        if np.linalg.norm(self._x_new-self._z_new) < 5e-2:
            self.converge = True
        
        
        #RHO
        if self._gamma * self._rho < self._rho_cap:
            self._delta_new = 1./96. * (np.linalg.norm(self._x_new - self._x_old)
                        +np.linalg.norm(self._z_new - self._z_old)
                        +np.linalg.norm(self._mu_new - self._mu_old))
            if self._delta_new > self._eta * self._delta_old:
                self._rho = self._gamma * self._rho
                self._alpha = 1. / ((self._sv+self._rho)*(1+1e-5))
            self._delta_old = self._delta_new
        
        np.copyto(self._z_old, self._z_new)
        np.copyto(self._mu_old, self._mu_new)
        np.copyto(self._x_old, self._x_new)


    def iterate(self, max_iter=20, plot_iter=1, verbose=False, plot_verbose=False):
        r"""Iterate

        This method calls update until either convergence criteria is met or
        the maximum number of iterations is reached

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default is ``150``)

        """

        for idx in range(max_iter+1):
            self._update(verbose=verbose)
            if plot_verbose and idx % plot_iter == 0:
                plt.figure(figsize=(16,5))
                plt.suptitle(str(idx))
                plt.subplot(142)
                plt.title('X')
                plt.imshow(self._x_old)
                plt.colorbar()
                plt.subplot(142)
                plt.title('input net')
                plt.imshow(self._z_tmp)
                plt.colorbar()
                plt.subplot(143)
                plt.title('Z')
                plt.imshow(self._z_old)
                plt.colorbar()
                plt.subplot(144)
                plt.title('HX-Y')
                plt.imshow(get_conv2d(self._x_old, self._psf, mode=self._convolve_mode, transp=False)-self._obs)
                plt.colorbar()
                plt.show()
            if self.converge:
                print(' - Converged after %d iterations'%(idx))
                break

        # rename outputs as attributes
        self.x_final = self._x_new
        self.z_final = self._z_new
        self.mu_final = self._mu_new


