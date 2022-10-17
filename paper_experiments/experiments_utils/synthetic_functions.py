import numpy as np

class BenchmarkFunction:
    
    def __init__(self, name, d):
        self.name = name
        self.d = d

    def __call__(self, x, z):
        pass
    
    
class StronglyConvexFunction(BenchmarkFunction):
    
    def __init__(self, d, seed =12):
        super().__init__("StronglyConvex", d)
        self.rnd_state = np.random.RandomState(seed = seed)
        self.A = np.random.rand(d, d)
        
    def get_exact_value(self, x):
        return 1/self.d * (np.linalg.norm(self.A.dot(x))**2)
        
    def __call__(self, x, z):
        row = self.A[z, :]
        return 1/self.d * (np.linalg.norm(row.dot(x))**2)
    
class PLConvexFunction(StronglyConvexFunction):
    
    def __init__(self, d, seed = 12):
        super().__init__(d, seed)
        self.name = "PLConvex" 
        U, D, VT = np.linalg.svd(self.A)
        D[-1] = 0.0
        self.A = (U @ np.diag(D) @ VT)
        
class PLNonConvexFunction(BenchmarkFunction):
    
    def __init__(self, d, seed = 12):
        super().__init__("PL Non-Convex", d)
        rnd_state = np.random.RandomState(seed=seed)
        A = rnd_state.randn(d, d)
        vh, s, u = np.linalg.svd(A, full_matrices=True)
        s[0] = 1.0
        s[1:] = 0.0
        self.A = vh.dot(np.diag(s)).dot(u)# np.dot(vh, np.dot(np.diag(s), u)).astype(np.float64)
       # print(self.A)
        self.c = rnd_state.rand(d).astype(np.float64) 
       # print("c: ", self.c)
       # print("Ac: ", self.A.dot(self.c))
        
    def get_exact_value(self, x):
        return np.square(np.linalg.norm(self.A.dot(x))) + 3*np.square(np.sin(self.c.dot(x))) #np.square(np.sin(self.c.dot(x)))
    
    def __call__(self, x, z):
        return np.square(np.linalg.norm(self.A[z,:].reshape(1, -1).dot(x))) + 3*np.square(np.sin(self.c.dot(x)))               
    