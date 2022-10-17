import numpy as np

from scipy.special import gammainc
from sszd.direction_matrices import DirectionStrategy

class GaussianStrategy(DirectionStrategy):
    
    def build_direction_matrix(self):
        return self.rnd_state.randn(self.d, self.l)

