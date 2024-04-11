import torch
from typing import Callable

class Optimizer:
    def __init__(self, name, target : Callable):
        self.name = name
        self.target = target

    def _build_result(self, x, fun_values = None, iterates = None, num_evals = None):
        result = {'x' : x}
        if fun_values is not None:
            result['fun_values'] = fun_values
        if iterates is not None:
            result['iterates'] = iterates
        if num_evals is not None:
            result['num_evals'] = num_evals
        return result


    def optimize(self, x0 : torch.Tensor, T : int, verbose : bool = False, return_trace : bool = False):
        raise NotImplementedError("This method is implemented in subclass of this class!")
        