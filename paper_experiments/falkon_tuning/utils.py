import torch
import numpy as np
import pandas as pd

from falkon import Falkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions

from sklearn.model_selection import train_test_split

MAXITER = 100

def build_model(param_config, M):
    kernel = GaussianKernel(torch.from_numpy(param_config[:-1]))
    penalty = param_config[-1]
    config ={
        'kernel' : kernel,
        'penalty' : penalty,
        'M' : M,
        'maxiter' : MAXITER,
        'options' : FalkonOptions(cg_tolerance=np.sqrt(1e-7))
    }
    return Falkon(**config)
    

    
    
    