import torch


class BenchmarkFunction:
    
    def __init__(self, name, d):
        self.name = name
        self.d = d
        
    def __repr__(self):
        return "{} [d = {}]".format(self.name, self.d)
        
        
class StronglyConvex(BenchmarkFunction):
    
    def __init__(self, d, seed = 1212, dtype = torch.float32, device = 'cpu'):
        super().__init__("StronglyConvex", d)
        self.generator = torch.Generator(device=device)
        self.dtype = dtype
        self.device = device
        self.generator.manual_seed(seed)
        self.A = self._build_fullrank()
    
    def _build_fullrank(self):
        A = None
        while A is None or torch.linalg.matrix_rank(A) != self.d:
            A = torch.rand(self.d, self.d, generator=self.generator, device=self.device, dtype=self.dtype)
        return A
    
    def __call__(self, x, z = None):
        if z is None:
            return torch.linalg.norm(self.A @ x.T, dim=0).square()
        return torch.linalg.norm(self.A[z, :].reshape(1, self.d) @ x.T, dim=0).square()
        

class PLConvexFunction(StronglyConvex):
    
    def __init__(self, d, seed = 1212, dtype = torch.float32, device = 'cpu'):
        super().__init__(d=d, seed=seed, dtype=dtype, device=device)
        self.name = "PLConvex" 
        U, D, VT = torch.linalg.svd(self.A)
        D[-2:] = 0.0
        self.A = (U @ torch.diag(D) @ VT)
        
class PLNonConvexFunction(BenchmarkFunction):
    
    def __init__(self, d, seed = 1212, dtype = torch.float32, device = 'cpu'):
        super().__init__("PL Non-Convex", d)
        self.generator = torch.Generator(device=device)
        self.dtype = dtype
        self.device = device
        self.generator.manual_seed(seed)
        self.A = self._build_matrix()
    
    def _build_matrix(self):
        A = torch.rand(self.d, self.d, generator=self.generator, device=self.device, dtype=self.dtype)
        vh, s, u = torch.linalg.svd(A, full_matrices=True)
        s[1:] = 0.0
        A = vh.matmul(torch.diag(s).matmul(u))
        self.c = torch.tensor([[0.150 for _ in range(self.d)]], dtype=self.dtype, device=self.device)
        return A
   
    def __call__(self, x, z = None):
        if z is None:
            return torch.linalg.norm(self.A @ x.T, dim = 0).square() + 3*torch.sin(2 * self.c @ x.T).square()         
        return torch.linalg.norm(self.A[z, :].reshape(-1, self.d) @ x.T, dim = 0).square() + 3*torch.square(torch.sin(self.c @ x.T)) 
