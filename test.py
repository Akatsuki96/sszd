
import numpy as np
from stozhopt import StoZhOpt
import time

fun = lambda x:x[0]**2 + x[1]**2 + x[2]**2


d = 40
l = 1

alpha = 1/7#lambda t : 1/np.sqrt(t)
h = lambda t : 1e-7



optimizer = StoZhOpt('spherical', d, l, alpha, 1e-7, dtype=np.float64, seed= 13)

opt_time = time.time()
optimizer.optimize(fun, np.array([9.0 for _ in range(d)], dtype=np.float64), 10, verbose=True)

print("[--] OPT TIME: {}".format(time.time() - opt_time))