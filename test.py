
import torch
import numpy as np
from stozhopt import StoZhOpt
from stozhopt.direction_strat import CoordinateDescentStrategy


fun = lambda x:x[0]**2 + x[1]**2

test_P = CoordinateDescentStrategy(2, 2, device="cpu",seed=13)


alpha = lambda t : 1/np.sqrt(t)

optimizer = StoZhOpt(test_P, alpha, 1e-5)

#    def optimize(self, fun, x0, T : int = 100, verbose : bool = False):

optimizer.optimize(fun, torch.tensor([-10.0, 5.0]), 1000, verbose=True)