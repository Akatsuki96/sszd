import torch
from typing import Callable, Optional, Any
from sszd.optimizers.opt import Optimizer
from sszd.direction_matrices.direction_matrix import DirectionMatrix


class DDS(Optimizer):

    def __init__(self, target : Callable[[torch.Tensor, Optional[Any]], torch.Tensor], 
                 alpha_0 : float,  
                 theta : float,
                 rho: float, 
                 ds_constant : float,
                 alpha_lims,
                 P : DirectionMatrix
                 ):
        super().__init__('dds', target)
        self.alpha = alpha_0
        self.alpha_lims = alpha_lims # (min_alpha, max_alpha)
        self.theta = theta # contraction factor
        self.rho = rho # expansion factor
        self.ds_constant = ds_constant
        self.P = P

    def forcing_fun(self, p : torch.Tensor):
        return min(1e-5, 1e-5 * (self.alpha**2) * p.norm(p=2).square().item())


    def optimize(self, x0: torch.Tensor, sample_z, T : int, verbose : bool = False, return_trace : bool = False):
        x_k = x0.clone()
        iterates, fun_values, num_evals = None, None, None
        if return_trace:
            iterates = [x_k.cpu().clone()]
            fun_values = [self.target(x_k)]
            num_evals = [1]
        k = 0
        nev = 0
        while k < T:
#            k += 1
            z_k = sample_z()
            f_k = self.target(x_k, z_k)
            P = self.P()
            self.D = torch.hstack((P, - P)) 
            for i in range(self.D.shape[1]):
                d = self.D[:, i].flatten()
                f_next = self.target(x_k + self.alpha * d, z_k)
                k += 1
                nev += 1
               # print(f_next - f_k)
#                print(self.alpha)
                if f_next - f_k <  - self.ds_constant * (self.alpha**2) * d.norm().square():#self.forcing_fun(d):
                    x_k = x_k + self.alpha * d
                    self.alpha = min(self.alpha * self.rho, self.alpha_lims[1])
                    if return_trace:
                        fun_values.append(self.target(x_k).item())
                        num_evals.append(nev)
                    nev = 0
                    break
                if k == T:
                    break

            else:
                self.alpha = max(self.alpha * self.theta, self.alpha_lims[0])
        if nev > 0 and return_trace:
            fun_values.append(self.target(x_k).item())
            num_evals.append(nev)
        return self._build_result(x_k, fun_values = fun_values, iterates=iterates, num_evals=num_evals)                  
