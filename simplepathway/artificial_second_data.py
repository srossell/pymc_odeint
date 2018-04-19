import numpy as np
from scipy.integrate import odeint

from artificialdata import init, time_vec, p_vec, v_in, odes, obs


factor = 0.5

np.random.seed(42)

xt2 = odeint(odes, init, time_vec, args=(p_vec, v_in*factor))

two_obs = np.concatenate((
        obs, 
        xt2.flatten() + np.random.randn(xt2.flatten().shape[0])*0.15
    ))

