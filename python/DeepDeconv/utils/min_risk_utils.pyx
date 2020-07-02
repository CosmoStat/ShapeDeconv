import scipy.optimize

import cython
from cython.parallel import prange
cimport numpy as cnp
import numpy as np
from libc.math cimport pow
from cpython cimport bool

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
cpdef cy_sure_proj_risk_est_1d(double mu, double[::1] psf_ps,double[::1] y_ps,  double[::1] reg_ps,
                               Py_ssize_t Ndata, double sigma2):
   
    """SURE estimate of projection risk for Tikhonov deconvolution.

    .. math:: 
        \| y - Hx \|^2 + \mu \| Lx \|^2
    
    :param mu: Regularisation parameter balancing l2 discrepancy and quadratic regularizer.
    :param psf_ps: 1D kernel power spectrum.
    :param y_ps: 1D noisy image power spectrum
    :param reg_ps: 1D regularization kernel power spectrum
    :param Ndata: number of frequencies in power spectrum.
    :param sigma2: white noise variance.
    :type mu: double 
    :type psf_ps: 1D npy.ndarray
    :type y_ps: 1D npy.ndarray
    :type reg_ps: 1D npy.ndarray
    :type Ndata: int_t
    :type sigma2: double

    :returns: SURE projection risk estimate
    :rtype: double

    .. warning:: the power spectra need be 1D
    .. warning:: mu must be positive (not checked)
    .. warning:: mu contains noise variance (:math:`\\mu = \\tau * \\sigma^2`)
    """

    cdef Py_ssize_t kx
    cdef double den=0.
    cdef double risk=0. 

    
    for kx in range(Ndata):
        den=psf_ps[kx]+mu*reg_ps[kx]
        risk+=psf_ps[kx]*y_ps[kx]/pow(den,2.0)+2.0*(sigma2-y_ps[kx])/den
    return risk


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
cpdef cy_sure_pred_risk_est_1d(double mu, double[::1] psf_ps, double[::1] y_ps, double[::1] reg_ps,
                               Py_ssize_t Ndata, double sigma2):
   
    """SURE estimate of prediction risk for Tikhonov deconvolution 

    .. math::
        \|y-Hx\|^2 + \\mu \|Lx\|^2
    
    :param mu: Regularisation parameter balancing l2 discrepancy and quadratic regularizer.
    :param psf_ps: 1D kernel power spectrum.
    :param y_ps: 1D noisy image power spectrum
    :param reg_ps: 1D regularization kernel power spectrum
    :param Ndata: number of frequencies in power spectrum.
    :param sigma2: white noise variance.
    :type mu: double 
    :type psf_ps: 1D npy.ndarray
    :type y_ps: 1D npy.ndarray
    :type reg_ps: 1D npy.ndarray
    :type Ndata: int_t
    :type sigma2: double

    :returns: SURE prediction risk estimate
    :rtype: double

    .. warning:: the power spectra need be 1D
    .. warning:: mu must be positive (not checked)
    .. warning:: mu contains noise variance (:math:`\\mu = \\tau * \\sigma^2` if :math:`\\tau` is the quadratic prior weight)
    """

    cdef Py_ssize_t kx
    cdef double wiener=0., wiener2=0.
    cdef double risk=0. 
    
    for kx in range(Ndata):
        wiener=psf_ps[kx]/(psf_ps[kx]+mu*reg_ps[kx])
        wiener2=pow(wiener,2.0)
        risk+=wiener2*y_ps[kx]+2*(sigma2-y_ps[kx])*wiener

    return risk


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
cpdef cy_gcv_risk_est_1d(double mu,double[::1] psf_ps, double[::1] y_ps, double[::1] reg_ps,
                               Py_ssize_t Ndata):
   
    """Generalized Cross Validation (GCV) for Tikhonov deconvolution 
    
    .. math::
        \|y-Hx\|^2 + \mu \|Lx\|^2
    
    :param mu: Regularisation parameter balancing l2 discrepancy and quadratic regularizer.
    :param psf_ps: 1D kernel power spectrum.
    :param y_ps: 1D noisy image power spectrum
    :param reg_ps: 1D regularization kernel power spectrum
    :param Ndata: number of frequencies in power spectrum.
    :type mu: double 
    :type psf_ps: 1D npy.ndarray
    :type y_ps: 1D npy.ndarray
    :type reg_ps: 1D npy.ndarray
    :type Ndata: int_t

    :returns: GCV risk estimate
    :rtype: double

    .. warning:: :math:`\\mu` power spectra need be 1D
    .. warning:: :math:`\\mu` must be positive (not checked)
    .. warning:: mu contains noise variance (:math:`\\mu = \\tau \\sigma^2` if :math:`\\tau` is the quadratic prior weight)

    
    """

    cdef Py_ssize_t kx
    cdef double wiener=0., wiener2=0.
    cdef double den=0., num=0.
    cdef double risk=0. 
    
    for kx in range(Ndata):
        wiener=psf_ps[kx]/(psf_ps[kx]+mu*reg_ps[kx])
        num+=y_ps[kx]*pow(1.0-wiener,2.0)
        den+=(1.0-wiener)
    return num/pow(den,2.0)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
cpdef cy_pereyra_hyper(double mu, double alpha, double beta, double[::1] psf_ps, 
                               double[::1] y_ps, double[::1] reg_ps,
                               Py_ssize_t Ndata,Py_ssize_t Nit, double sigma2,bool marg=False ):
   
    """MAP or marginal MAP Tikhonov hyperparameter selection using Hierarchical Bayesian method with gamma prior for hyperparameter
       
    .. math::
        \\frac{1}{2} \|y-Hx\|^2 + \\frac{1}{2} \\tau \\sigma^2 \|Lx\|^2 - (\\alpha - 1+ \\frac{n}{k}) \\log \\tau + \\beta \\tau
    
    :param mu: initial hyperparamer value.
    :param alpha: shape of gamma distribution.
    :param beta: rate of gamma distribution.
    :param psf_ps: 1D kernel power spectrum.
    :param y_ps: 1D noisy image power spectrum
    :param reg_ps: 1D regularization kernel power spectrum
    :param Ndata: number of frequencies in power spectrum.
    :param Nit: number of iterations to estimate MAP.
    :param sigma2: white noise variance.
    :param marg: white noise variance. default is False
    :type mu: double 
    :type alpha: double 
    :type beta: double 
    :type psf_ps: 1D npy.ndarray
    :type y_ps: 1D npy.ndarray
    :type reg_ps: 1D npy.ndarray
    :type Ndata: int_t
    :type Nit: int_t
    :type sigma2: double

    :returns: MAP or marginalized MAP Hyperparameter 
    :rtype: double

    .. warning:: the power spectra need be 1D
    .. warning:: :math:`\\mu` must be positive (not checked)
    .. warning:: To obtain quadratic weigh :math:`\\mu`, one need to do :math:`\\mu=\\frac{1}{2}*\\tau*\\sigma^2`.
    """

    cdef Py_ssize_t kx,kit
    cdef double deconvf2=0.
    cdef double hyp_cur=mu
    cdef double num_alpha=alpha-1
    
    if marg:
        num_alpha=alpha 
    
    for kit in range(Nit):
        deconvf2=0
        for kx in range(Ndata):
            deconvf2+=psf_ps[kx]*reg_ps[kx]*y_ps[kx]/pow(psf_ps[kx]+hyp_cur*sigma2*reg_ps[kx],2.0)
        hyp_cur=(Ndata/2.0 + num_alpha)/(0.5*deconvf2+beta)
    return hyp_cur


def min_risk_est_1d(psf_ps,y_ps,reg_ps,sigma2,method,risktype="SureProj",mu0=1.0):
    
    """Tikhonov hyperparameter selection using various risk minimization strategies
           
    :param psf_ps: 1D kernel power spectrum.
    :param y_ps: 1D noisy image power spectrum
    :param reg_ps: 1D regularization kernel power spectrum
    :param Ndata: number of frequencies in power spectrum.
    :param Nit: number of iterations to estimate MAP.
    :param sigma2: white noise variance.
    :param method: Minimization strategy
    :param risktype: risk to minimize
    :param mu0: Initial hyperparamer balancing likelihood and quadratic prior.
    :type psf_ps: 1D npy.ndarray
    :type y_ps: 1D npy.ndarray
    :type reg_ps: 1D npy.ndarray
    :type Ndata: int_t
    :type Nit: int_t
    :type sigma2: double
    :type method: string among "Powell", "Brent" or "Golden" or "Bounded"
    :type risktype: string among "SureProj","SurePred" or "GCV"
    :type mu0: double 

    :returns: Optimization from from scipy.optimize
    :rtype: OptimizeResult

    .. warning:: the power spectra need be 1D
    .. warning:: mu0 must be positive (not checked)
    .. warning:: mu0 contains noise variance (:math:`\\mu = \\tau * \\sigma^2` if :math:`\\tau` is the quadratic prior weight)
    .. seealso:: scipy.optimize.minimize, scipy.optimize.minimize_scalar
    """
    
    bounds=scipy.optimize.Bounds(1e-6,np.inf,keep_feasible=True)
    bounds_1d=(1e-6,100000)
    if(risktype == "SureProj"):
        if method == "Powell":
            return scipy.optimize.minimize(cy_sure_proj_risk_est_1d, mu0, args=(psf_ps,y_ps,reg_ps, y_ps.size,sigma2), method='Powell',
                             bounds=bounds,options={'xtol': 1e-4, 'maxiter': 1000, 'disp': False})
        elif method == "Brent" or method == "Golden":
            return scipy.optimize.minimize_scalar(cy_sure_proj_risk_est_1d, args=(psf_ps,y_ps,reg_ps, y_ps.size,sigma2), method=method,
                             bounds=bounds_1d,options={'xtol': 1e-4, 'maxiter': 1000})
        elif method == "Bounded":
            return scipy.optimize.minimize_scalar(cy_sure_proj_risk_est_1d, args=(psf_ps,y_ps,reg_ps, y_ps.size,sigma2), method=method,
                             bounds=bounds_1d,options={'xatol': 1e-4, 'maxiter': 1000})
        else:
            raise ValueError("Optim. Method {0} is not supported".format(method))
    elif(risktype == "SurePred"):
        if method == "Powell":
            return scipy.optimize.minimize(cy_sure_pred_risk_est_1d, mu0, args=(psf_ps,y_ps,reg_ps, y_ps.size,sigma2), method='Powell',
                             bounds=bounds,options={'xtol': 1e-4, 'maxiter': 1000, 'disp': False})
        elif method == "Brent" or method == "Golden" :
            return scipy.optimize.minimize_scalar(cy_sure_pred_risk_est_1d, args=(psf_ps,y_ps,reg_ps, y_ps.size,sigma2), method=method,
                             bounds=bounds_1d,options={'xtol': 1e-4, 'maxiter': 1000})
        elif method == "Bounded":
            return scipy.optimize.minimize_scalar(cy_sure_pred_risk_est_1d, args=(psf_ps,y_ps,reg_ps, y_ps.size,sigma2), method=method,
                             bounds=bounds_1d,options={'xatol': 1e-4, 'maxiter': 1000})
        else:
            raise ValueError("Optim. Method {0} is not supported".format(method))

    elif(risktype == "GCV"):
        if method == "Powell":
            return scipy.optimize.minimize(cy_gcv_risk_est_1d, mu0, args=(psf_ps,y_ps,reg_ps, y_ps.size), method='Powell',
                             bounds=bounds,options={'xtol': 1e-4, 'maxiter': 1000, 'disp': False})
        elif method == "Brent" or method == "Golden":
            return scipy.optimize.minimize_scalar(cy_gcv_risk_est_1d, args=(psf_ps,y_ps,reg_ps, y_ps.size), method=method,
                             bounds=bounds_1d,options={'xtol': 1e-4, 'maxiter': 1000})
        elif method == "Bounded":
            return scipy.optimize.minimize_scalar(cy_gcv_risk_est_1d, args=(psf_ps,y_ps,reg_ps, y_ps.size), method=method,
                             bounds=bounds_1d,options={'xatol': 1e-4, 'maxiter': 1000})
        else:
            raise ValueError("Optim. Method {0} is not supported".format(method))
    else:
        raise ValueError("Risk {0} is not supported".format(risktype))
        
