import pymc

# load the model
import pk_model as mod

# fit the model with mcmc
mc = pymc.MCMC(mod)
mc.use_step_method(pymc.AdaptiveMetropolis, [mod.kcp, mod.kpc, mod.ke])
mc.sample(iter=2000, burn=1000, thin=20, verbose=1)
