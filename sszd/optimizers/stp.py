import torch
import tqdm
import numpy as np 
from typing import Callable, Optional, Any
from sszd.optimizers.opt import Optimizer
from sszd.direction_matrices.unstructured_directions import SphericalDirections, GaussianDirections


class STP(Optimizer):

    def __init__(self, target : Callable[[torch.Tensor, Optional[Any]], torch.Tensor], d : int,  alpha : Callable[[int], float],  dtype = torch.float32, device : str = 'cpu', seed : int = 121314):
        super().__init__('stp', target)
        self.alpha = alpha
        self.P = SphericalDirections(d = d, l = 1, seed =seed, dtype = dtype, device=device)

    def _approx_grad(self, x : torch.Tensor, z, h : float):
        P_k = self.P()
        grad = torch.zeros((x.shape[0], ), dtype = x.dtype, device = x.device)
        fx = self.target(x, z)
        for i in range(P_k.shape[1]):
            grad += (self.target(x + h * P_k[:, i], z) - fx) * P_k[:, i]
        return grad.div_(h)

    def optimize(self, x0: torch.Tensor, sample_z, T : int, verbose : bool = False, return_trace : bool = False):
        x_k = x0.clone()
        iterates, fun_values, num_evals = None, None, None
        if return_trace:
            iterates = [x_k.cpu().clone()]
            fun_values = [self.target(x_k)]
            num_evals = [1]
        k = 0
        iters = 0
        while k < T:
            alpha_k = self.alpha(iters)
            z_k = sample_z()
            g = self.P()[:, 0]
            points = [x_k - alpha_k * g, x_k, x_k + alpha_k * g]
            values = [self.target(x, z_k).item() for x in points ]
            x_k = points[np.argmin(values)]
            if return_trace:
                iterates.append(x_k.cpu().clone())
                fun_values.append(self.target(x_k).item())
                num_evals.append(3)
            iters += 1
            k += 3
        return self._build_result(x_k, fun_values = fun_values, iterates=iterates, num_evals=num_evals)                  
