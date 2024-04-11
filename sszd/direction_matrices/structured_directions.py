import torch
from sszd.direction_matrices import DirectionMatrix

from math import sqrt

class QRDirections(DirectionMatrix):
    
    def __init__(self, normalize = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nrm_const = sqrt(self.d / self.l) if normalize else 1.0

    def __call__(self):
        A = torch.randn(size=(self.d, self.l), generator=self.generator, device=self.device, dtype=self.dtype)
        return  self.nrm_const * torch.linalg.qr(A, mode='reduced')[0]
        
class RandomCoordinate(DirectionMatrix):
    

    def __init__(self, normalize = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inds = torch.ones(self.d, device=self.device)
        self.nrm_const = sqrt(self.d / self.l) if normalize else 1.0
    
    def __call__(self):
        P = torch.zeros((self.d, self.l), device=self.device, dtype=self.dtype)
        inds = self.inds.multinomial(self.l, replacement=False, generator=self.generator)
        P[inds, range(self.l)] = ((torch.rand(self.l, device=self.device, generator=self.generator) < 0.5)*2 - 1).to(dtype=self.dtype) 
        return self.nrm_const * P
    
class RandomHouseholder(DirectionMatrix):

    def __init__(self, normalize = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.I = torch.eye((self.d, self.l), dtype=self.dtype, device=self.device)
        self.nrm_const = sqrt(self.d / self.l) if normalize else 1.0

    def __call__(self):
        v = torch.randn((self.d, ), generator=self.generator, device=self.device, dtype=self.dtype)
        v.div_(v.norm(p=2))
        return self.nrm_const * (self.I - 2 * v.outer(v[:self.l]))