import torch
import numpy as np

from .dir_strat import DirectionStrategy 

class SphericalSmoothingStrategy(DirectionStrategy):

    def __init__(self, d, l=1, device="cpu", dtype=torch.float32, seed=None):
        super().__init__(d, l, device, dtype, seed)
    
    def build_direction_matrix(self):
        Zk = torch.randn(size=(self.d, self.l), generator=self.rnd_state, dtype=self.dtype, device=self.device)
        Q_k, R_k = torch.linalg.qr(Zk, mode="complete")
        diag_R_k = torch.diag(R_k)

        D = (diag_R_k / torch.linalg.norm(diag_R_k.reshape(-1,1), dim=1)) * torch.eye(self.d, self.l, device=self.device)
        del Zk
        if self.device == "cuda":
             torch.cuda.empty_cache()
        return  np.sqrt(self.d / self.l) * Q_k.matmul(D)