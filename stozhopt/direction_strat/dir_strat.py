import torch


class DirectionStrategy:

    def __init__(self, d, l = 1, device = "cpu", dtype=torch.float32, seed=None):
        self.d = d
        self.l = l 
        self.device = device
        self.dtype = dtype
        self.rnd_state = torch.Generator(device = device)
        self.rnd_state.seed() if seed is None else self.rnd_state.manual_seed(seed)

    def build_direction_matrix(self):
        pass