import torch
from typing import Callable, Optional, Any
from sszd.optimizers.opt import Optimizer
from sszd.direction_matrices.direction_matrix import DirectionMatrix


class SVRZ(Optimizer):

    def __init__(self, target : Callable[[torch.Tensor, Optional[Any]], torch.Tensor], 
                 n, 
                 m,
                 alpha : float, 
                 h : float,# Callable[[int], float], 
                 P : DirectionMatrix, G : DirectionMatrix):
        super().__init__('svrz', target)
        self.alpha = alpha
        self.h = h
        self.m = m
        self.n = n
        self.P = P
        self.G = G

    def _approx_grad(self, x : torch.Tensor, z, h : float):
        P_k = self.P()
        grad = torch.zeros((x.shape[0], ), dtype = x.dtype, device = x.device)
        fx = self.target(x, z)
        for i in range(P_k.shape[1]):
            grad += (self.target(x + h * P_k[:, i], z) - fx) * P_k[:, i]
        return (x.shape[0] / P_k.shape[1]) * grad.div_(h)

    def _approx_full_grad(self, x: torch.Tensor, h : float):
        P_k = self.G()
        grad = torch.zeros((x.shape[0], ), dtype = x.dtype, device = x.device)
        fx = self.target(x)
        for i in range(P_k.shape[1]):
            grad += (self.target(x + h * P_k[:, i]) - fx) * P_k[:, i]
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
            g_k = self._approx_full_grad(x_k, self.h)
            k += (self.G.l + 1) * self.n
            x_tau = x_k.clone()
            for i in range(self.m):
                z_k = sample_z()
                g_i = self._approx_grad(x = x_tau, z = z_k, h = self.h)
                g_j = self._approx_grad(x = x_k, z = z_k, h=self.h)
                x_k = x_k - self.alpha * (g_i - g_j + g_k)
                k += 2 * (self.P.l + 1)
                if k >= T:
                    break
            if return_trace:
                iterates.append(x_k.cpu().clone())
                fun_values.append(self.target(x_k).item())
                num_evals.append(2 * (self.P.l + 1) * self.m + (self.G.l + 1) * self.n)
            iters += 1
        return self._build_result(x_k, fun_values = fun_values, iterates=iterates, num_evals=num_evals)                  
