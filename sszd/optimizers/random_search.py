import torch
import numpy as np 
from typing import Callable, Optional, Any
from sszd.optimizers.opt import Optimizer
from sszd.direction_matrices.unstructured_directions import SphericalDirections, GaussianDirections


class RandomSearch(Optimizer):

    def __init__(self, target : Callable[[torch.Tensor, Optional[Any]], torch.Tensor], sigma : float = 1.0,  dtype = torch.float32, device : str = 'cpu', seed : int = 121314):
        super().__init__('random_search', target)
        self.dtype=dtype
        self.device=device
        self.sigma = sigma
        self.gen = torch.Generator(device)
        self.gen.manual_seed(seed)

    def optimize(self, x0: torch.Tensor, sample_z, T : int, verbose : bool = False, return_trace : bool = False):
        x_k = x0.clone()
        k = 0
        iterates, fun_values, num_evals = None, None, None
        if return_trace:
            iterates = [x_k.cpu().clone()]
            fun_values = [self.target(x_k)]
            num_evals = [1]
        current_best = (x_k, self.target(x_k, sample_z()))
        while k < T:
            x_new = x_k + self.sigma * torch.randn(size=x0.shape, generator=self.gen, dtype=self.dtype, device=self.device)
            z_k = sample_z()
            f_new = self.target(x_new, z_k)
            if f_new < current_best[1]:
                current_best = (x_new, f_new)
            if return_trace:
                iterates.append(x_k.cpu().clone())
                fun_values.append(self.target(current_best[0]).item())
                num_evals.append(1)
            k += 1
        return self._build_result(x_k, fun_values = fun_values, iterates=iterates, num_evals=num_evals)                  
