import numpy as np
from .dir_strat import DirectionStrategy


class CoordinateDescentStrategy(DirectionStrategy):

    def __init__(self, d, l = 1, dtype= np.float32, seed = None):
        super().__init__(d, l, dtype, seed)
        self.I = np.eye(d, dtype=dtype)


    def build_direction_matrix(self):
        P = self.I[:, self.rnd_state.choice(self.d, size=self.l,replace=False)]
        
        mask = (self.rnd_state.rand(self.l) < 0.5).astype(int)
        mask[mask==0] = -1
    
        return P * mask