import pymc as pm
import numpy as np
from scipy.integrate import odeint

from artificialdata import init
from artificialdata import obs
from artificialdata import odes
from artificialdata import time_vec
from artificialdata import v_in
from artificial_second_data import factor
from artificial_second_data import two_obs

Vmax = pm.Uniform('Vmax', 0., 100.)
Km = pm.Uniform('Km', 0., 10.)

# Deterministic model
@pm.deterministic
def mymodel(p0=Vmax, p1=Km):
    sol_one = odeint(odes, init, time_vec, args=([p0, p1], v_in)).flatten()
    sol_two = odeint(odes, init, time_vec, args=([p0, p1], v_in * factor))\
                .flatten()
    xt = np.concatenate((sol_one, sol_two))
    return xt

xt = pm.Lambda('xt', lambda mymodel=mymodel: mymodel)

# data likelihood
A = pm.Normal('A', mu=xt, tau=1, value=two_obs, observed=True)

