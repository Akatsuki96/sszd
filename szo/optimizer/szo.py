from typing import Callable
import numpy as np
from szo.direction_strat import CoordinateDescentStrategy, SphericalSmoothingStrategy


def str_strategy(dir_build, d, l, dtype, seed):
    if dir_build=='coordinate':
        return CoordinateDescentStrategy(d, l=l, dtype=dtype, seed=seed)
    elif dir_build=='spherical':
        return SphericalSmoothingStrategy(d, l=l,  dtype=dtype, seed=seed)
    raise Exception("dir_build should be 'coordinate', 'spherical' or an extension of DirectionStrategy")

def get_strategy(dir_build, d, l, dtype, seed):
    if type(dir_build) == type:
        return dir_build(d, l=l, dtype=dtype, seed=seed)
    elif isinstance(dir_build, str):
        return str_strategy(dir_build, d, l, dtype, seed)
    raise Exception("dir_build should be 'coordinate', 'spherical' or an extension of DirectionStrategy")



class SZO:

    def __init__(self, dir_build, d, l,  alpha, h,  dtype = np.float32, seed : int = 12):

        self.dir_build = get_strategy(dir_build, d, l, dtype, seed)
        self.d, self.l = d, l
        self.dtype = dtype
        self.seed= seed
        self.alpha = alpha
        self.h, self.t = h, 1

    def get_alpha(self, t):
        if not isinstance(self.alpha, Callable):
            return self.alpha
        return self.alpha(t)

    def get_h(self, t):
        if not isinstance(self.h, Callable):
            return self.h
        return self.h(t)

    def optimize(self, fun, x0, T : int = 100, verbose : bool = False):
        x = x0#.to(self.device)
        fx = fun(x)
        if verbose:
            print("[--] t: 0\tx: {}\tf(x): {}".format(x, fx))
        for t in range(1, T+1):
            P_k = self.dir_build.build_direction_matrix()
            h_k = self.get_h(t)
            alpha_k = self.get_alpha(t)
            grad = 0 
            for i in range(self.l):
                grad += ((fun(x + P_k[:,i] * h_k) - fx)/h_k) * P_k[:, i]

            x = x - alpha_k * grad
            fx = fun(x)
            if verbose:
                print("[--] t: {}\tx: {}\tf(x): {}".format(t, x, fx))
            
        return x
    
    def stochastic_step(self, fun, x, batch, *args):
        fx = fun(x, batch, *args)
        P_k = self.dir_build.build_direction_matrix()
        h_k = self.get_h(self.t)
        alpha_k = self.get_alpha(self.t)

        grad = np.zeros(self.l)
        for i in range(self.l):
            grad[i] = ((fun(x + P_k[:,i] * h_k, batch, *args) - fx)/h_k) #* P_k[:, i]
        self.t+=1
        return x - alpha_k * P_k.dot(grad), grad, fx


    def step(self, fun, x, verbose = False):
        fx = fun(x)
        if verbose:
            print("[--] f(x): {}".format(fx))
        P_k = self.dir_build.build_direction_matrix()
        h_k = self.get_h(self.t)
        alpha_k = self.get_alpha(self.t)

        grad = 0 
        for i in range(self.l):
        #    print("\tfx: {}\t P_k[:,i] * h_k: {}\tf: {}".format(fx, x+ P_k[:,i] * h_k,fun(x + P_k[:,i] * h_k) ))
            grad += ((fun(x + P_k[:,i] * h_k) - fx)/h_k) * P_k[:, i]
        #print("[--] grad: {}".format(grad))
        self.t+=1
        return x - alpha_k * grad, grad


    def reset(self):
        self.t = 1