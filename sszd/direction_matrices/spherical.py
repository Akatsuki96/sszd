import numpy as np

from .dir_strat import DirectionStrategy 

class SphericalStrategy(DirectionStrategy):

    def __init__(self, d, l=1, dtype=np.float32, seed=None):
        super().__init__(d, l, dtype, seed)
    
    def build_direction_matrix(self):
        Zk = self.rnd_state.randn(self.d, self.l).astype(np.float64)
        

        Q_k, R_k = np.linalg.qr(Zk, mode="complete")

        diag_R_k = np.diag(R_k)
    
        D = (diag_R_k / np.abs(diag_R_k)) * np.eye(self.d, self.l, dtype=self.dtype)
        
        P_k =  np.sqrt(self.d / self.l) * Q_k.dot(D)

        return P_k