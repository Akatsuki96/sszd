import os
import torch 
import numpy as np

import matplotlib.pyplot as plt


def build_data(N, d, m1, m2, std1, std2, seed=121212, dtype=torch.float64, device='cpu'):
    data_generator = torch.Generator(device=device)
    data_generator.manual_seed(seed)
    
    data = torch.empty((2*N, d), device=device, dtype=dtype)

    for i in range(N):
        data[i] = torch.normal(mean=m1, std=std1, generator=data_generator)

    y1 = torch.zeros(N, dtype=dtype, device=device)
    y2 = torch.ones(N, dtype=dtype, device=device)
    
    for i in range(N, 2*N):
        data[i] = torch.normal(mean=m2, std=std2, generator=data_generator)
    
    return data, torch.hstack((y1, y2))    

def build_dataset(N=1000, d = 100, device='cpu', dtype=torch.float64):

    m1 = torch.tensor([5.0 for _ in range(d)], device=device, dtype=dtype)
    m2 = torch.tensor([0.0 for _ in range(d)], device=device, dtype=dtype)
    s1 = torch.tensor([0.4 for _ in range(d)], device=device, dtype=dtype)
    s2 = torch.tensor([0.4 for _ in range(d)], device=device, dtype=dtype)

    X, y = build_data(N, d, m1, m2, s1, s2, device=device, dtype=dtype)
    return X, y


def store_result(results, out):
    os.makedirs("{}".format(out), exist_ok = True)
    mean, std, ct_mean, ct_std = results
    with open("{}/mean_value.log".format(out), "w") as f:
        for i in range(mean.shape[0]):
            f.write("{},{}\n".format(mean[i], std[i]))
    with open("{}/ctime.log".format(out), "w") as f:
        for i in range(ct_mean.shape[0]):
            f.write("{},{}\n".format(ct_mean[i], ct_std[i]))