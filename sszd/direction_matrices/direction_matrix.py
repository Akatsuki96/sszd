import torch
from math import sqrt

class DirectionMatrix:
    
    def __init__(self, d, l, seed=None, dtype=torch.float32, device='cpu'):
        self.d = d
        self.l = l
        self.dtype = dtype
        self.device = device
        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(seed)
        
        
    def __call__(self):
        raise NotImplementedError("Abstract class representing a random matrix generator")
    