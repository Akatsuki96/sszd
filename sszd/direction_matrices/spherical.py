import numpy as np

from .dir_strat import DirectionStrategy 

class SphericalStrategy(DirectionStrategy):

    def __init__(self, d, l=1, dtype=np.float32, seed=None):
        super().__init__(d, l, dtype, seed)
        self.I = np.eye(self.d, self.l, dtype=self.dtype)
    
    def build_direction_matrix(self):
        Zk = self.rnd_state.randn(self.d, self.l).astype(np.float64)
        Q_k, _ = np.linalg.qr(Zk, mode="complete")
        return np.sqrt(self.d / self.l) * Q_k.dot(self.I)

