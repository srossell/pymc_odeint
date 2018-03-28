"""
Very simple metabolic model
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

t = np.arange(0, 10, 0.5)
actual_k1 = 9
actual_k2 = 1

def odes(x, t, k1, k2):
    return k1 - k2*x[0]

xt = odeint(odes, [0], t, args=(actual_k1, actual_k2,))
xt_with_noise = xt.flatten() + 0.5*np.random.randn(len(xt))

if __name__ == '__main__':
    plt.plot(t, xt, 'ko', label='odeint')
    plt.plot(t, xt_with_noise, 'r.', label='with noise')
    plt.legend()
    plt.show()
