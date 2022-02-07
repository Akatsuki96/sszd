import torch

from stozhopt.direction_strat import DirectionStrategy

class StoZhOpt:

    def __init__(self, dir_build : DirectionStrategy, alpha, h, device : str="cpu"):
        self.dir_build = dir_build
        self.alpha = alpha
        self.device = device
        self.h = h
        self.t = 1

    @property
    def l(self):
        return self.dir_build.l

    def get_alpha(self, t):
        if type(self.alpha) == float:
            return self.alpha
        return self.alpha(t)

    def get_h(self, t):
        if type(self.h) == float:
            return self.h
        return self.h(t)

    

    def compute_finite_diff(self, fun, x, fx, p_i, h):
        print("[--] x + p_i * self.h: {}".format(x + p_i * h))
        print("[--] x + p_i * self.h: {}".format( fun(x + p_i * h)))
        
        return (fun(x + p_i * h) - fx)/h * p_i

    def optimize(self, fun, x0, T : int = 100, verbose : bool = False):
        x = x0.to(self.device)
        fx = fun(x)
        if verbose:
            print("[--] t: 0\tx: {}\tf(x): {}".format(x, fx))
        for t in range(1, T+1):
            P_k = self.dir_build.build_direction_matrix()
            h_k = self.get_h(t)
            alpha_k = self.get_alpha(t)
            print("[--] alpha_k: {}\th_k: {}".format(alpha_k, h_k))

            grad = 0 
            for i in range(self.l):
                grad = grad + (fun(x + P_k[:,i] * h_k) - fx)/h_k * P_k[:, i]

            x = x - alpha_k * grad
            fx = fun(x)
            if verbose:
                print("[--] t: {}\tx: {}\tf(x): {}\tgrad: {}".format(t, x, fx, grad))
            
        return x