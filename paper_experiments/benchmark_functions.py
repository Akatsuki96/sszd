import numpy as np

class BenchmarkFunction:
    
    def __init__(self, name, d, f_star):
        self.name = name
        self.d = d
        self.f_star = f_star
        
    def __call__(self, x):
        pass
    
    def complete(self, x, theta):
        pass
        
    def __repr__(self):
        return "{} [d = {}]".format(self.name, self.d)
    
    
class SquareNorm(BenchmarkFunction):
    def __init__(self, d, rnd_state = np.random.RandomState(42)):
        assert d > 1
        super().__init__("SquareNorm", d, 0.0)
        self.rnd_state = rnd_state
        self.A = rnd_state.rand(d, d).astype(np.float64)
        
    def complete(self, x):
        return 1/self.d * np.square(np.linalg.norm(self.A.dot(x))**2)
    
    def __call__(self, x, theta):
        # theta is a row of A
        return 1/self.d * np.square(np.linalg.norm(self.A[theta,:].reshape(1, -1).dot(x))**2)               
    
class SquareNormPL(BenchmarkFunction):
    def __init__(self, d, rnd_state = np.random.RandomState(42)):
        assert d > 1
        super().__init__("SquareNormPL", d, 0.0)
        self.rnd_state = rnd_state
        A = rnd_state.rand(d, d)
        vh, s, u = np.linalg.svd(A, full_matrices=False)
        s[rnd_state.randint(d)] = 0.0 # set an eigenvalue to 0.0
        self.A = np.dot(vh, np.dot(np.diag(s), u)).astype(np.float64)
        
    def complete(self, x):
        return 1/self.d * np.square(np.linalg.norm(self.A.dot(x))**2)
    
    def __call__(self, x, theta):
        # theta is a row of A
        return 1/self.d * np.square(np.linalg.norm(self.A[theta,:].reshape(1, -1).dot(x))**2)               
    
class NonConvPL(BenchmarkFunction):
    def __init__(self, d, rnd_state = np.random.RandomState(42)):
        assert d > 1
        super().__init__("PL", d, 0.0)
        self.rnd_state = rnd_state
        A = rnd_state.rand(d, d)
        vh, s, u = np.linalg.svd(A, full_matrices=False)
        s[0] = 1.0
        s[1:] = 0.0
        self.A = np.dot(vh, np.dot(np.diag(s), u)).astype(np.float64)
        self.c = rnd_state.rand(d).astype(np.float64) 
        
    def complete(self, x):
        return 1/self.d * np.square(np.linalg.norm(self.A.dot(x))**2) + 3*np.square(np.sin(self.c.T.dot(x)))
    
    def __call__(self, x, theta):
        # theta is a row of A
        return 1/self.d * np.square(np.linalg.norm(self.A[theta,:].reshape(1, -1).dot(x))**2) + 3*np.square(np.sin(self.c.T.dot(x)))               
    
    

class Rosenbrock(BenchmarkFunction):
    def __init__(self, d):
        assert d > 1
        super().__init__("Rosenbrock", d, 0.0)
        
    def complete(self, x):
        fx = 0
        for i in range(self.d - 1):
            fx += 100*np.square(x[i+1] - np.square(x[i])) + np.square(x[i] - 1)
        return fx
    
    def __call__(self, x, theta):
        # theta is an index in [0, 1]
        assert theta == 0 or theta == 1
        if theta == 0:
            fx = 0
            for i in range(self.d - 1):
                fx += 100*np.square(x[i+1] - np.square(x[i]))
            return fx
        fx = 0
        for i in range(self.d - 1):
            fx += np.square(x[i] - 1)
        return fx
        
        