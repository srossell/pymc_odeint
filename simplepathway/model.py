import pymc as pm
import numpy as np
from scipy.integrate import odeint

from artificialdata import init
from artificialdata import obs
from artificialdata import odes
from artificialdata import p_vec
from artificialdata import time_vec
from artificialdata import v_in

Vmax = pm.Uniform('Vmax', 0., 100.)
Km = pm.Uniform('Km', 0., 10.)

init = [0.]

# Deterministic model
@pm.deterministic
def mymodel(p0=Vmax, p1=Km):
    soln = odeint(odes, init, time_vec, args=([p0, p1], v_in))
    xt = soln.flatten()
    return xt
xt = pm.Lambda('xt', lambda mymodel=mymodel: mymodel)

# data likelihood
A = pm.Normal('A', mu=xt, tau=1, value=obs, observed=True)

