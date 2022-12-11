import torch
from sszd.direction_matrices import DirectionMatrix

class RandomCoordinate(DirectionMatrix):
    
    def __init__(self, mul_seed=12, mul_device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.mgen = torch.Generator(device=mul_device)
        self.mgen.manual_seed(mul_seed)
        self.inds = torch.ones(self.d, device=mul_device)
    
    
    def __call__(self):
        P = torch.zeros((self.d, self.l), device=self.device, dtype=self.dtype)
        inds = self.inds.multinomial(self.l, replacement=False, generator=self.mgen)
        P[inds, range(self.l)] = ((torch.rand(self.l, device=self.device, generator=self.generator) < 0.5)*2 - 1).to(dtype=self.dtype) 
        return self.nrm_const * P