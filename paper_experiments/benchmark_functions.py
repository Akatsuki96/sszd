import torch
from math import sqrt
from typing import Any


class LeastSquares:

    def __init__(self, n, d, L, mu, x_star = None, dtype = torch.float32, device='cpu', seed = 121314) -> None:
        assert x_star is None or x_star.shape[0] == d
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.n = n
        self.d = d
        A = torch.randn((n, d), generator=self.generator, dtype=dtype, device=device)
        U, s, V = torch.linalg.svd(A)
        s = torch.linspace(sqrt(L), sqrt(mu), s.shape[0], dtype=dtype, device=device)
        self.A = U @ torch.diag(s) @ V

        self.y = self.A @ x_star
    
    def __call__(self, x, z = None) -> Any:
        if z is None:
            return (self.A @ x - self.y).norm(p=2).square()
        return (self.A[z, :] @ x - self.y[z]).norm().square()
    
class PLConvex(LeastSquares):

    def __init__(self, n, d, L,  x_star = None, dtype = torch.float32, device = 'cpu', seed = 121314) -> None:
        super().__init__(n=n, d=d, L=L, mu=0.0, x_star=x_star, dtype=dtype, device=device, seed=seed)
    
    
class PLNonConvex:
    
    def __init__(self, d, L, mu, dtype = torch.float32, device='cpu', seed = 121314) -> None:
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.d = d
        A = torch.randn((d, d), generator=self.generator, dtype=dtype, device=device)
        U, s, V = torch.linalg.svd(A, full_matrices=True)
        s[0] = sqrt(L)
        s[1:] = 0.0
        self.A = U @ torch.diag(s) @ V
        self.c = torch.tensor([0.150 for _ in range(self.d)], dtype=dtype, device=device)

    def __call__(self, x, z = None):
        if z is None:
            return (self.A @ x).norm(p=2).square() + 3*torch.sin(2 * self.c @ x).square()         
        return (self.A[z, :] @ x).norm(p=2).square() + 3*torch.square(torch.sin(self.c @ x)) 


# class LogisticRegression:



#     def build_data(N, d, m1, m2, std1, std2, seed=121212, dtype=torch.float64, device='cpu'):
#     data_generator = torch.Generator(device=device)
#     data_generator.manual_seed(seed)
    
#     data = torch.empty((2*N, d), device=device, dtype=dtype)

#     for i in range(N):
#         data[i] = torch.normal(mean=m1, std=std1, generator=data_generator)

#     y1 = torch.zeros(N, dtype=dtype, device=device)
#     y2 = torch.ones(N, dtype=dtype, device=device)
    
#     for i in range(N, 2*N):
#         data[i] = torch.normal(mean=m2, std=std2, generator=data_generator)
    
#     return data, torch.hstack((y1, y2))    

# def build_dataset(N=1000, d = 100, device='cpu', dtype=torch.float64):

#     m1 = torch.tensor([2.0 for _ in range(d)], device=device, dtype=dtype)
#     m2 = torch.tensor([0.0 for _ in range(d)], device=device, dtype=dtype)
#     s1 = torch.tensor([0.1 for _ in range(d)], device=device, dtype=dtype)
#     s2 = torch.tensor([0.4 for _ in range(d)], device=device, dtype=dtype)

#     X, y = build_data(N, d, m1, m2, s1, s2, device=device, dtype=dtype)
#     return X, y

# N = 500
# d = 20
