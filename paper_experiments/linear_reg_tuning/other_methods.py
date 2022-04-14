import numpy as np

class Optimizer:

    def stochastic_step(self, fun, x, batch):
        pass
    
    def reset(self):
        pass
    
class GDSOptions:
    
    def __init__(self, d, alpha_max=10.0, exp_factor=1., 
                 cont_factor=0.5, gen_strat="compass", 
                 sketch = None, forcing_fun = lambda x: x**2):
        self.d = d
        self.alpha_max = alpha_max
        self.exp_factor = exp_factor
        self.cont_factor = cont_factor
        self.gen_strat = gen_strat
        self.forcing_fun = forcing_fun
        self.sketch = sketch
        
    def generate_gset(self):
        if self.gen_strat == "compass":
            G = np.vstack((np.eye(self.d), -np.eye(self.d)))
        elif self.gen_strat == "np1":
            G = np.vstack((np.eye(self.d), -np.ones((self.d,1))))
        elif self.gen_strat == "random_unit":
            a = np.random.normal(size=(self.d,1))
            a = a / np.linalg.norm(a)
            G = np.vstack([a, -a])
        elif self.gen_strat == "random_orth":
            A = np.random.normal(size=(self.d,self.d))
            Q = np.linalg.qr(A, mode='reduced')[0] 
            G = np.vstack([Q, -Q])
        else:
            raise Exception("Unknown generation strategy!")
        if self.sketch is None:
            return G
        # Sketch G
    
    
class GDS(Optimizer):
    def __init__(self, init_alpha, options:GDSOptions):
        self.init_alpha = init_alpha
        self.alpha=init_alpha
        self.options = options
        self.t = 1
        
    @property
    def alpha_max(self):
        return self.options.alpha_max
    
    @property
    def cont_factor(self):
        return self.options.cont_factor
    
    @property
    def exp_factor(self):
        return self.options.exp_factor
    
    @property
    def forcing_fun(self):
        return self.options.forcing_fun
    
    def stochastic_step(self, fun, x, batch):
        self.D = self.options.generate_gset()
        values = np.zeros(self.D.shape[0])
        fx = fun(x, batch)
        while True:
            for i in range(self.D.shape[0]):
                value = fun(x + self.alpha * self.D[i], batch)
                if value < fx - self.forcing_fun(self.alpha):
                    self.alpha = min(self.alpha * self.exp_factor, self.alpha_max)
                    return x + self.alpha*self.D[i], 1
            self.alpha *= self.cont_factor
        
    def reset(self):
        self.t=1
        self.alpha=self.init_alpha
    

