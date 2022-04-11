import numpy as np

class Optimizer:

    def stochastic_step(self, fun, x, batch):
        pass
    
    def reset(self):
        pass

class CompassSearch(Optimizer):
    
    def __init__(self, dim, init_alpha=1.0):
        self.D = np.vstack((np.eye(dim), -np.eye(dim)))
        self.init_alpha=init_alpha
        self.alpha = init_alpha
        self.t=1
    
    def stochastic_step(self, fun, x, batch):
        values = np.zeros(self.D.shape[0])
        fx = fun(x, batch)
        while True:
            for i in range(self.D.shape[0]):
                values[i] = fun(x + self.alpha * self.D[i], batch)
            best_idx = np.argmin(values)
            if values[best_idx] < fx - self.alpha**2:
                return x + self.alpha*self.D[best_idx], 1
            self.alpha /= 2
        
    def reset(self):
        self.t=1
        self.alpha=self.init_alpha