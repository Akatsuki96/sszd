import torch
from typing import Callable, Optional, Any
from sszd.optimizers.opt import Optimizer
from sszd.direction_matrices.direction_matrix import DirectionMatrix


class SSZD(Optimizer):

    def __init__(self, target : Callable[[torch.Tensor, Optional[Any]], torch.Tensor], alpha : Callable[[int], float], h : Callable[[int], float], P : DirectionMatrix):
        super().__init__('sszd', target)
        self.alpha = alpha
        self.h = h
        self.P = P

    def _approx_grad(self, x : torch.Tensor, z, h : float):
        P_k = self.P()
        grad = torch.zeros((x.shape[0], ), dtype = x.dtype, device = x.device)
        fx = self.target(x, z)
        for i in range(P_k.shape[1]):
            grad += (self.target(x + h * P_k[:, i], z) - fx) * P_k[:, i]
        return grad.div_(h)

    def optimize(self, x0: torch.Tensor, sample_z, T : int, verbose : bool = False, return_trace : bool = False):
        x_k = x0.clone()
        k = 0
        iters = 0
        iterates, fun_values, num_evals = None, None, None
        if return_trace:
            iterates = [x_k.cpu().clone()]
            fun_values = [self.target(x_k)]
            num_evals = [1]
        while k < T:
            alpha_k = self.alpha(iters)
            h_k = self.h(iters)
            z_k = sample_z()
            g = self._approx_grad(x = x_k, z = z_k, h = h_k)
            x_k = x_k -alpha_k * g
            if return_trace:
                iterates.append(x_k.cpu().clone())
                fun_values.append(self.target(x_k).item())
                num_evals.append(self.P.l + 1)
            if verbose:
                print(self.target(x_k).item(), g.norm().item())
            k += self.P.l + 1
            iters += 1
        return self._build_result(x_k, fun_values = fun_values, iterates=iterates, num_evals=num_evals)                  
