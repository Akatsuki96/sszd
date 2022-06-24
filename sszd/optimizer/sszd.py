import numpy as np
from typing import Callable
from sszd.utils import get_strategy

class SSZD:

    def __init__(self, dir_strategy, d, l,  alpha, h,  dtype = np.float32, seed : int = 12, bounds : np.ndarray = None):
        self.dir_strategy = get_strategy(dir_strategy, d, l, dtype, seed)
        self.d, self.l = d, l
        self.alpha = alpha
        self.h = h
        self.dtype = dtype
        self.seed = seed
        self.bounds = bounds
        self.t = 1

    def get_alpha(self, t):
        if not isinstance(self.alpha, Callable):
            return self.alpha
        return self.alpha(t)

    def get_h(self, t):
        if not isinstance(self.h, Callable):
            return self.h
        return self.h(t)
    
    def set_l(self, l):
        assert 0 < l <= self.d
        self.l = l
        self.dir_strategy.l = l

    def _clip_value(self, x):
        if self.bounds is not None:
            return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
        return x

    def step(self, fun, x, *args):
        fx = fun(x, *args)
        P_k = self.dir_strategy.build_direction_matrix()
        h_k = self.get_h(self.t)
        alpha_k = self.get_alpha(self.t)

        grad = 0 
        for i in range(self.l):
            x_d = self._clip_value(x + P_k[:,i] * h_k)
            grad += ((fun(x_d, *args) - fx)/h_k) * P_k[:, i]
        
        self.t+=1
        return self._clip_value(x - alpha_k * grad), grad


    def reset(self):
        self.t = 1