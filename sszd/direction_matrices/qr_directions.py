import torch
from sszd.direction_matrices import DirectionMatrix

class StructuredSphericalDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_truncated_Q(self, A):    
        return torch.linalg.qr(A, mode='reduced')[0]
    
    def __call__(self):
        A = torch.randn(size=(self.d, self.l), generator=self.generator, device=self.device, dtype=self.dtype)
        return self.nrm_const * self._compute_truncated_Q(A)
    
    
        
class GaussianDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self):
        return torch.randn(size=(self.d, self.l), generator=self.generator, device=self.device, dtype=self.dtype)
       
       
class SphericalDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self):
        A = torch.randn(size=(self.d, self.l), generator=self.generator, device=self.device, dtype=self.dtype)
        A /= torch.linalg.norm(A, 2, dim=0)
        return A
       