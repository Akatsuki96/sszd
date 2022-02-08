import torch
from .dir_strat import DirectionStrategy


class CoordinateDescentStrategy(DirectionStrategy):

    def __init__(self, d, l = 1, device = "cpu", dtype= torch.float32, seed = None):
        super().__init__(d, l, device, dtype, seed)
        self.I = torch.eye(d, dtype=dtype)


    def build_direction_matrix(self):
        P = self.I[:, torch.randperm(self.d, generator=self.rnd_state, dtype=torch.long)[:self.l]]
        P.mul_(2*(torch.rand(self.l, generator=self.rnd_state, dtype=torch.float32) < 0.5) - 1).to(self.device)
        return P