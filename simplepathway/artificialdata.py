"""
Very simple metabolic model
"""

import numpy as np
from scipy.integrate import odeint

np.random.seed(42)

init = [0.]
time_vec = np.arange(0, 10, 1)
p_vec = [10, 1]
v_in = 9

def odes(x, t, p, v_in):
    Vmax = p[0]
    Km = p[1]
    return np.array([
        v_in - Vmax * x[0] / (Km + x[0]),
        ])

xt = odeint(odes, init, time_vec, args=(p_vec, v_in))
obs = xt.flatten() + np.random.randn(xt.flatten().shape[0])*0.15

