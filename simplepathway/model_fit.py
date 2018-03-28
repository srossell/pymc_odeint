import pymc
import pandas as pd
import matplotlib.pyplot as plt

# load the model
import model as mod

# fit the model with mcmc
mc = pymc.MCMC(mod)
mc.use_step_method(pymc.AdaptiveMetropolis, [mod.Vmax, mod.Km])
mc.sample(iter=20000, burn=1000, thin=20, verbose=1)

# get samples
Vmax_sample = mc.trace('Vmax')[:]
Km_sample = mc.trace('Km')[:]

df = pd.DataFrame({'Vmax': Vmax_sample, 'Km': Km_sample})
pd.plotting.scatter_matrix(df)
plt.show()

