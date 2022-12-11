import torch
import numpy as np

class DirectionMatrix:
    
    def __init__(self, d, l, seed=None, dtype=torch.float32, device='cpu'):
        self.d = d
        self.l = l
        self.dtype = dtype
        self.device = device
        self.generator = torch.Generator(device=device)
        self.mseed = None
        self.nrm_const = np.sqrt(d/l)
        if seed is not None:
            self.generator.manual_seed(seed)
            self.mseed = seed
        
    @property
    def seed(self):
        if self.mseed is not None:
            return self.mseed
        return self.generator.initial_seed()
        
    def __call__(self):
        raise NotImplementedError("Abstract class representing a random matrix generator")
    