import pymc as pm
import numpy as np
from scipy.integrate import odeint

from create_artificial_data import xt_with_noise as obs
from create_artificial_data import t as tspan

#time = np.arange(0, 10, 1)

k1 = pm.Uniform('k1', 0., 100.)
k2 = pm.Uniform('k2', 1., 100.)

#tspan = time
init = [0.]

# Deterministic model
@pm.deterministic
def mymodel(k1=k1, k2=k2):
    def odes(x, t):
        return k1 - k2*x[0]
    xt = odeint(odes, init, tspan)
    return xt[:, 0]

xt = pm.Lambda('xt', lambda mymodel=mymodel: mymodel)

# data likelihood
A = pm.Normal('A', mu=xt, tau=1, value=obs, observed=True)

