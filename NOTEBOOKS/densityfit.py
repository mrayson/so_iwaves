"""
Least-squares fit a profile of N2 to data

Example 

```
    ii = 0
    N2in, zin = N2[ii,:], -np.log(z[ii,:])
    idx = ~np.isnan(N2in)
    N2max = 5e-4

    initguess = [1e-5/N2max,  3e-4/N2max, 5.4, 1, 5.5, 1, 0.5]
    bounds = [ (1e-6/N2max, 1e-5/N2max, 1, 0.1, 1.4, 0.1, 0.01), (1e-5/N2max, 2/N2max, 7, 6, 7, 6, 0.99)]

    N2fit, f0, err = fit_rho_lsq(N2in[idx]/N2max, zin[idx], double_gaussian_N2_v2, bounds, initguess)

    plt.figure()
    plt.plot(N2in, zin,'-')
    plt.plot(N2fit*N2max, zin[idx])
```
"""
from scipy.optimize import minimize
import numpy as np


def double_gaussian_N2(z, beta):
    return beta[0,...] + beta[1,...] * (np.exp(- ((z+beta[2,...])/beta[3,...])**2 )  +\
              np.exp(-((z+beta[4,...])/beta[5,...])**2 ) )

def double_gaussian_N2_v2(z, beta):
    w1 = beta[6]
    w2 = 1-w1
    return beta[0,...] + beta[1,...] * (w1*np.exp(- ((z+beta[2,...])/beta[3,...])**2 )  +\
              w2*np.exp(-((z+beta[4,...])/beta[5,...])**2 ) )


def rho_err(coeffs, rho, z, density_func):
    """
    Returns the difference between the estimated and actual data
    """
    
    soln = density_func(z, coeffs)

    err =  np.linalg.norm(rho - soln)
    return err

def fit_rho(rho, z, density_func, bounds, initguess):
    """
    Fits an analytical density/N^2 profile to data
    Uses a robust linear regression
    Inputs:
    ---
        rho: vector of density (or N^2) [Nz]
        z : depth [Nz] w/ negative values i.e. 0 at surface, positive: up
        density_func, bounds, initguess: 
    Returns:
    ---
        rhofit: best fit function at z locations
        f0: tuple with analytical parameters
        err: L2-norm of the error vector
    """
    status = 0

    H = np.abs(z).max()   

    soln =\
        minimize(rho_err, 
                 initguess, 
                 args=(rho, z, density_func), \
                 method='nelder-mead',
                bounds=bounds,    
        )
    f0 = soln['x']

    rhofit = density_func(z, f0)
    
    return rhofit, f0, soln