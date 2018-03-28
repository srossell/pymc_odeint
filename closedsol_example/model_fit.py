import pymc
import pandas as pd
import matplotlib.pyplot as plt

# load the model
import model as mod

# fit the model with mcmc
mc = pymc.MCMC(mod)
mc.use_step_method(pymc.AdaptiveMetropolis, [mod.k1, mod.k2])
mc.sample(iter=20000, burn=1000, thin=2, verbose=1)
#mc.sample(40000, 10000, 1)
# get samples
k1_sample = mc.trace('k1')[:]
k2_sample = mc.trace('k2')[:]

df = pd.DataFrame({'k1': k1_sample, 'k2': k2_sample})
pd.plotting.scatter_matrix(df)
plt.show()

