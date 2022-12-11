from sszd.direction_matrices import DirectionMatrix

class Optimizer:
    def __init__(self, P : DirectionMatrix, alpha, h):
        self.P = P        
        self.alpha = alpha
        self.h = h
        self.t = 1
        
    def step(self, fun, x, *args):
        raise NotImplementedError("Abstract class!")
    
    def reset(self):
        self.t = 1
        