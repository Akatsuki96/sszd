import torch

from stozhopt.direction_strat import DirectionStrategy, CoordinateDescentStrategy, SphericalSmoothingStrategy


def str_strategy(dir_build, d, l, device, dtype, seed):
    if dir_build=='coordinate':
        return CoordinateDescentStrategy(d, l=l, device=device, dtype=dtype, seed=seed)
    elif dir_build=='spherical':
        return SphericalSmoothingStrategy(d, l=l, device=device, dtype=dtype, seed=seed)
    raise Exception("dir_build should be 'coordinate', 'spherical' or an extension of DirectionStrategy")

def get_strategy(dir_build, d, l, device, dtype, seed):
    if type(dir_build) == type:
        return dir_build(d, l=l, device=device, dtype=dtype, seed=seed)
    elif isinstance(dir_build, str):
        return str_strategy(dir_build, d, l, device, dtype, seed)
    raise Exception("dir_build should be 'coordinate', 'spherical' or an extension of DirectionStrategy")



class StoZhOpt:

    def __init__(self, dir_build, d, l,  alpha, h, device : str="cpu", dtype = torch.float32, seed : int = 12):

        self.dir_build = get_strategy(dir_build, d, l, device, dtype, seed)
        self.d, self.l = d, l
        self.dtype = dtype
        self.seed= seed
        self.alpha = alpha
        self.device = device
        self.h, self.t = h, 1

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

            grad = 0 
            for i in range(self.l):
                grad += ((fun(x + P_k[:,i] * h_k) - fx)/h_k) * P_k[:, i]

            x = x - alpha_k * grad
            fx = fun(x)
            if verbose:
                print("[--] t: {}\tx: {}\tf(x): {}\tgrad: {}".format(t, x, fx, grad.numpy()))
            
        return x