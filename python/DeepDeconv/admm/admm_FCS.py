#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 2019

@author: fsureau
"""
import os, sys
if os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import numpy as np
import matplotlib.pylab as plt

from modopt.opt.algorithms import SetUp
from modopt.opt.cost import costObj

from DeepDeconv.utils.conv_utils import get_conv2d_fftconv
from DeepDeconv.utils.data_utils import max_sv
from skimage import restoration


class Cost_Lagrangian_deconv(object):
    def cost(self, *args):
        x, z, mu, rho, obs, psf,var = args[0]
        cost_val = 0.5 * rho * np.linalg.norm(z-x) ** 2 - np.dot(mu.flatten(), (z-x).flatten()) + 0.5/var * np.linalg.norm(obs - get_conv2d_fftconv(x, psf)) ** 2
        return cost_val

class ADMM_deconv(SetUp):
    def __init__(self, x, z, mu, obs, psf, sigma, dnn,
             rho=50, gamma=1.4, eta=0.5, rho_cap=200,
             opti_method='FISTA', convolve_mode='scipy_fft',quad_prior=0.):
        """ADMM deconvolution.

        :param X: 2D deconvolved image
        :param Z: 2D image for variable after denoising 
        :param mu: 2D image containing dual parameter
        :param obs: noisy observed image
        :param psf: point spread function associated to image, that should be centered
        :param sigma: noise standard deviation
        :param dnn: denoiser NN
        :param rho: augmented lagrangian hyperparameter
        :param eta: minimal decrease of global residual to keep rho constant
        :param gamma: increase of rho when minimal decrease not reached
        :param rho_cap: maximal value for rho
        :param opti_method: string among "TIKHONOV","GD" or "FISTA" to set the inversion step
        :param convolve_mode: string among "scipy_fft","scipy" to set the algorithm for convolution
        :type X: 2D npy.ndarray 
        :type Z: 2D npy.ndarray 
        :type mu: 2D npy.ndarray 
        :type obs: 2D npy.ndarray 
        :type psf: 2D npy.ndarray 
        :type sigma: double
        :type dnn: DeepDeconv.DeepNet.DeepNet object
        :type rho: double
        :type eta: double
        :type gamma: double
        :type rho_cap: double
        :type opti_method: string
        :type convolve_mode: string

        .. warning:: :math:`\\rho`,:math:`\\rho_{cap}`,:math:`\\eta`,:math:`\\gamma` must be positive (not checked)
        .. note:: :math:`\\rho=\\gamma\\rho` if :math:`\\Delta>\\eta\\Delta_{old}` and  if :math:`\\rho*\\gamma<\\rho_{}`
        """
        SetUp.__init__(self)
        #Minimizing: 1/(2*sigma**2) ||obs-psf*X||^2_2 + \lambda R(Z) + mu^T (X-Z)+rho/2||X-Z||^2_2
        #OR
        #Minimizing: 1/(2*sigma**2) ||obs-psf*X||^2_2 + \lambda R(Z) +rho/2||X-Z+mu/rho||^2_2
        # Set the initial variable values
        (self._check_input_data(data) for data in (x, z, mu))
        self._x_old = np.copy(x)
        self._z_old = np.copy(z)
        self._mu_old = np.copy(mu)    
        self._psf = psf
        # Set the algorithm operators
        self._obs = obs
        self._cost_func = costObj([Cost_Lagrangian_deconv()], tolerance=1e-10, verbose=False)
        self._delta_old = 0
        
        # Set the algorithm parameters
        (self._check_param(param) for param in (rho, gamma))
        self._rho= rho #augmented lagrangian hyperparameter
        self._gamma = gamma #multiplicative rho factor
        self._eta = eta #min residual factor If ∆k+1 ≥ η∆k, then ρk+1 = γρk.
        self._sigma = sigma # noise level
        self._rho_cap = rho_cap #max rho value
        self._sv = max_sv(psf) #||H||^2_2
        
        self._opti_method = opti_method
        self._convolve_mode = convolve_mode
        
        if opti_method=="TIKHONOV":            
            ext_size=np.array(psf.shape)+np.array(obs.shape)-1
            self.psf_ext=np.zeros(ext_size)
            self.obs_ext=np.zeros(ext_size)
            sz_psf=psf.shape
            sz_obs=obs.shape
            roll_ifft=ext_size//2-np.array(np.shape(psf))//2 #This is the shift to center the extended psf if the psf is originally centered when using ir2tf
            sl_ctr_ext=(slice(roll_ifft[0],roll_ifft[0]+sz_psf[0]),slice(roll_ifft[1],roll_ifft[1]+sz_psf[1]))
            self.psf_ext[sl_ctr_ext]=psf
            self.obs_ext[0:sz_obs[0],0:sz_obs[1]]=obs
            self._trans_func = restoration.uft.ir2tf(self.psf_ext,self.psf_ext.shape, is_real=False)
            self._hth_fft=np.abs(self._trans_func)**2
            self._obs_fft= restoration.uft.ufft2(self.obs_ext) 
            self._quad_prior=quad_prior
        else:
            # Save H.T*H and H.T*y
            self._hth = get_conv2d_fftconv(psf, psf, mode=self._convolve_mode, transp=True)
            self._hty = get_conv2d_fftconv(obs, psf, mode=self._convolve_mode, transp=True)
            self._alpha = 1. / ((self._sv/(self._sigma**2)+self._rho)*(1+1e-5))
        
        # Set the Neural Network
        self._DNN = dnn
        
        
    
    def _x_gradient_descent(self, max_iter=1000, stop_criterion=1e-6):
        x_temp = np.copy(self._x_old)
        for i in range(max_iter):
            grad = get_conv2d_fftconv(x_temp, self._hth, mode=self._convolve_mode, transp=False) - self._hty
            x_new = x_temp - self._alpha * (grad + self._mu_old + self._rho * (x_temp - self._z_old))
            x_new[x_new<0] = 0
            if np.linalg.norm(x_temp - x_new) < stop_criterion:
                print(i)
                return x_new
            np.copyto(x_temp, x_new)
        return x_new
    
    def _fista(self, max_iter=500):
        x_fista_old = np.copy(self._x_old)
        z_fista_old = np.copy(self._z_old)
        t_fista_old = 1
        for i in range(max_iter):
            grad = get_conv2d_fftconv(z_fista_old, self._hth, mode=self._convolve_mode, transp=False) - self._hty
            x_fista_new = z_fista_old - self._alpha * (grad + self._mu_old * self._sigma**2 + self._rho * self._sigma**2 * (z_fista_old - self._z_old))
            x_fista_new[x_fista_new<0] = 0
            t_fista_new = (1. + np.sqrt(4. * t_fista_old**2 + 1.))/2.
            lambda_fista = 1 + (t_fista_old - 1)/t_fista_new
            z_fista_new = x_fista_old + lambda_fista * (x_fista_new - x_fista_old)
            np.copyto(x_fista_old, x_fista_new)
            np.copyto(z_fista_old, z_fista_new)
            t_fista_old = t_fista_new
        return x_fista_old
    
    def _tikho(self):
        z_ext=np.zeros_like(self.obs_ext)
        sz_z=self._z_old.shape
        z_ext[0:sz_z[0],0:sz_z[1]]=self._z_old 
        mu_ext=np.zeros_like(self.obs_ext)
        mu_ext[0:sz_z[0],0:sz_z[1]]=self._mu_old 
        lag_add_fft= (self._sigma**2)*restoration.uft.ufft2(self._rho*z_ext + mu_ext)
        hfstar=np.conj(self._trans_func)
        filter_f=1.0/(self._hth_fft+(self._rho+self._quad_prior)*(self._sigma**2))
        x_ext=np.real(restoration.uft.uifft2(filter_f*(hfstar*self._obs_fft+lag_add_fft)))
        return x_ext[0:sz_z[0],0:sz_z[1]]
        
            
    def _dnn_process(self, input_dnn):
        out_dnn = self._DNN.model.predict(input_dnn.reshape((1,96,96,1)), batch_size=1, verbose=0)
        return out_dnn.reshape((96,96))
        
    def _update(self, verbose=True):
        #X
        if self._opti_method == 'GD':
            self._x_new = self._x_gradient_descent()
        elif self._opti_method == 'FISTA':
            self._x_new = self._fista()
        elif self._opti_method == 'TIKHONOV':
            self._x_new = self._tikho()

        
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
                plt.imshow(get_conv2d_fftconv(self._x_old, self._psf, mode=self._convolve_mode, transp=False)-self._obs)
                plt.colorbar()
                plt.show()
            if self.converge:
                print(' - Converged after %d iterations'%(idx))
                break

        # rename outputs as attributes
        self.x_final = self._x_new
        self.z_final = self._z_new
        self.mu_final = self._mu_new


