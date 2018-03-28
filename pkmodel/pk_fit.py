import pymc
import pandas as pd
import matplotlib.pyplot as plt

# load the model
import pk_model as mod

# fit the model with mcmc
mc = pymc.MCMC(mod)
mc.use_step_method(pymc.AdaptiveMetropolis, [mod.kcp, mod.kpc, mod.ke])
mc.sample(iter=50000, burn=10000, thin=2, verbose=1)

df = pd.DataFrame({'kcp': mc.trace('kcp')[:], 'kpc': mc.trace('kpc')[:],
                    'ke': mc.trace('ke')[:]})
pd.plotting.scatter_matrix(df)
plt.show()
