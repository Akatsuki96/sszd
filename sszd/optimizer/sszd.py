import torch
from sszd.optimizer.opt import Optimizer

class SSZD(Optimizer):
            
    def __approx_grad__(self, fun, x, h, *args):
        P = self.P()
        grad = torch.zeros((1, x.shape[1]), dtype = x.dtype, device = x.device)
        fx = fun(x, *args)
        for i in range(P.shape[1]):
            grad += (fun(x + h * P[:, i], *args) - fx) * P[:, i]
        return grad.div_(h)
        
    def step(self, fun, x, *args):
        alpha = self.alpha(self.t)
        h = self.h(self.t)
        d = self.__approx_grad__(fun, x, h, *args)
        self.t += 1
        return x - alpha * d