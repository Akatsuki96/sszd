
import numpy as np
from stozhopt import StoZhOpt
  



d = 4
l = 4

rnd_state = np.random.RandomState(12)

def rosenbrock(x):
    sm = 0
    for i in range(d-1):
        sm += 100*((x[i+1] - x[i]**2)**2) +(x[i] - 1)**2
    return sm

mx = 10
mn = -5
T = 10#0000

alpha = lambda t: 1e-5

optimizer = StoZhOpt("spherical", d=d, l=l, alpha=alpha, h=1e-5, dtype=np.float64, seed=12)

optimizer.optimize(rosenbrock, rnd_state.rand(d) * (mx - mn) + mn, T, verbose=True)

