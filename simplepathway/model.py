import pymc as pm
import numpy as np
from scipy.integrate import odeint

time = np.arange(0, 10, 1)

obs = np.array([1.096, 4.184, 3.195, 5.626, 5.832, 7.460, 8.107, 6.426, 7.563,
                8.212])

Vmax = pm.Uniform('Vmax', 0., 100.)
Km = pm.Uniform('Km', 0., 10.)

tspan = time
y0 = [0.]

# Deterministic model
@pm.deterministic
def mymodel(p0=Vmax, p1=Km):
    def odes(y, t):
        return ([
                    9 - p0 * y[0] / (p1 + y[0]),
                ])
    soln = odeint(odes, y0, tspan)
    xt = soln.flatten()
    return xt
xt = pm.Lambda('xt', lambda mymodel=mymodel: mymodel)

prec = pm.Gamma('precision', alpha=0.1, beta=0.1)

# data likelihood
A = pm.Normal('A', mu=xt, tau=1, value=obs, observed=True)


