
import torch
import numpy as np
from stozhopt import StoZhOpt
import time

fun = lambda x:x[0]**2 + x[1]**2 + x[2]**2


d = 3
l = 2

alpha = 0.5#lambda t : 1/np.sqrt(t)
h = lambda t : 1e-7



optimizer = StoZhOpt('spherical', d, l, alpha, 1e-5, device="cpu", dtype=torch.float32, seed= 13)

opt_time = time.time()
optimizer.optimize(fun, torch.tensor([-10.0, 5.0, 1.0]), 10, verbose=True)

print("[--] OPT TIME: {}".format(time.time() - opt_time))