import numpy as np


class DirectionStrategy:

    def __init__(self, d, l = 1, dtype=np.float32, seed=None):
        self.d = d
        self.l = l 
        self.dtype = dtype
        self.rnd_state = np.random.RandomState(seed)

    def build_direction_matrix(self):
        pass