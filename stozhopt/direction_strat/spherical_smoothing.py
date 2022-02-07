import torch
import numpy as np

from .dir_strat import DirectionStrategy 

class SphericalSmoothingStrategy(DirectionStrategy):

    def __init__(self, d, l=1, device="cpu", dtype=torch.float32, seed=None):
        super().__init__(d, l, device, dtype, seed)
    
    def build_direction_matrix(self):
        Zk = torch.randn(size=(self.d, self.d), generator=self.rnd_state, dtype=self.dtype, device=self.device)
        Qk, _ = torch.linalg.qr(Zk, mode="complete")
        del Zk
        if self.device == "cuda":
             torch.cuda.empty_cache()
        return  Qk[:,:self.l]