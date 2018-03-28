"""
Very simple metabolic model
"""

import numpy as np
from scipy.integrate import odeint

init = [0.]
time_vec = np.arange(0, 10, 1)
p_vec = [10, 1]

def odes(x, t, p):
    Vmax = p[0]
    Km = p[1]
    return np.array([
        9 - Vmax * x[0] / (Km + x[0]),
        ])

xt = odeint(odes, init, time_vec, args=(p_vec,))

# xt.flatten() + np.random.randn(xt.flatten().shape[0])
obs = [1.096, 4.184, 3.195, 5.626, 5.832, 7.460, 8.107, 6.426, 7.563, 8.212]
