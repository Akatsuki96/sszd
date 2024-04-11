import torch
from sszd.direction_matrices import DirectionMatrix


class GaussianDirections(DirectionMatrix):
    
    def __call__(self):
        return torch.randn(size=(self.d, self.l), generator=self.generator, device=self.device, dtype=self.dtype)
       
       
class SphericalDirections(DirectionMatrix):
    
    def __call__(self):
        A = torch.randn(size=(self.d, self.l), generator=self.generator, device=self.device, dtype=self.dtype)
        A /= torch.linalg.norm(A, dim=0)
        return A
