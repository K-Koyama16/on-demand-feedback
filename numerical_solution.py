# Purpose:  In this program, define functions to calculate the neuronal model.
#
# Author:   K.Koyama
#
# Function:
#   - runge_kutta
#           Runge-Kutta method of the fourth order for calculating differential equations
#   - stochastic_euler
#           Euler-Maruyama method


# Import of python libraries
import numpy as np
import numba
from numba import jit


@jit(nopython=True)
def runge_kutta(x, t, tau, params, model):
    """ 4th Runge-Kutta Method 
    Args:
        x (np.ndarray 1-dim): Variables
        t (float): Time
        tau (float): TimeStep
        params (np.ndarray 1-dim): Parameters in model
        model (function): Model<HR, Chay, ...>
    Returns:
        x_tmp (np.ndarray): Variables after calculation
    """
    x_tmp = np.zeros(len(x))

    half_tau = tau / 2 

    k1 = model(x, params) 
    
    x_tmp = x + half_tau * k1
    k2 = model(x_tmp, params)
    
    x_tmp = x + half_tau * k2
    k3 = model(x_tmp, params)
    
    x_tmp = x + tau * k3
    k4 = model(x_tmp, params)
    
    x_tmp = x + tau / 6 * (k1 + k4 + 2 * (k2 + k3)) 

    return x_tmp


@jit(nopython=True)
def stochastic_euler(x, tau, model, params, noise):
    x_new = x + model(x, params)*tau + np.sqrt(tau)*noise*np.ones(len(x))
    return x_new

