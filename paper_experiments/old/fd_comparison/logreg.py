import sys
import torch
import time

import numpy as np
from sszd import SSZD
from sszd.stepsize import StepSize
from sszd.direction_matrices import RandomCoordinate, StructuredSphericalDirections, SphericalDirections, GaussianDirections

sys.path.append("../")
sys.path.append("../other_methods")
from utils import *
from other_methods import *
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
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

    m1 = torch.tensor([2.0 for _ in range(d)], device=device, dtype=dtype)
    m2 = torch.tensor([0.0 for _ in range(d)], device=device, dtype=dtype)
    s1 = torch.tensor([0.1 for _ in range(d)], device=device, dtype=dtype)
    s2 = torch.tensor([0.4 for _ in range(d)], device=device, dtype=dtype)

    X, y = build_data(N, d, m1, m2, s1, s2, device=device, dtype=dtype)
    return X, y

N = 500
d = 20

X_full, Y_full = build_dataset(N=N, d = d,device=device)

X_full = (X_full - X_full.mean(dim=0))/X_full.std(dim=0)

rnd_state = np.random.RandomState(12)

inds = rnd_state.choice(X_full.shape[0], X_full.shape[0], replace=False)
X_full = X_full[inds, :]
Y_full = Y_full[inds]



def loss(h, y):
    return (-y * torch.log(h) - (1 - y) * torch.log(1 - h)).mean()

def target(w, z = None):
    if z is None:
        return loss(torch.sigmoid(X_full @ w.T), Y_full)

    x_p, y_p = z[0], z[1]
    return loss(torch.sigmoid(x_p @ w.T), y_p)

seed=121212
dtype = torch.float64
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'


def target(w, z = None):
    if z is None:
        X, Y = X_full, Y_full
    else:
        X, Y = z
    return ((X @ w.T) - Y).square().mean()


def run_sszd_experiment(target, init_x, optimizer, T, reps):
    dtype = torch.float64
    device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
    d, l = optimizer.P.d, optimizer.P.l
    results = [] 
    ctime = []
    seed = 121212


    stoc_gen = torch.Generator()
    stoc_gen.manual_seed(seed)
    

    for r in range(reps):

        x = torch.full((1, d), init_x, dtype=dtype).to(device)
        y_full = target(x).item()
        results.append([y_full for _ in range(l)])
        ctime.append([0.0 for _ in range(l)])
        for t in range(T//l - 1):
            it_time = time.time()
            idx = torch.randint(low=0, high=X_full.shape[0],size=(1,),generator=stoc_gen).item()
            z = (X_full[idx, :], Y_full[idx])
            x = optimizer.step(target, x, z)
            y_full = target(x).item()
            it_time = time.time() - it_time
            print("[SSZD] reps: {}/{}\tt: {}/{}\ty: {}".format(r, reps, t, T//l - 1, y_full))
            ctime[r].append(it_time)
            results[r].append(y_full)
            for _ in range(l-1):
                results[r].append(y_full)
                ctime[r].append(0.0)
        optimizer.reset()
    ctime = np.cumsum(ctime, 1)
    return (np.mean(results, axis=0), np.std(results, axis=0), np.mean(ctime, axis=0), np.std(ctime, axis=0))



T = 5000
reps = 3

P1_c = RandomCoordinate(d=d, l=d, seed=seed, dtype=dtype, device=device)
P1_s = StructuredSphericalDirections(d=d, l=d, seed=seed, dtype=dtype, device=device)
P_nest = GaussianDirections(d=d, l=1, seed=seed, dtype=dtype, device=device)
P_duchi = GaussianDirections(d=d, l=d, seed=seed, dtype=dtype, device=device)
P_flax = SphericalDirections(d=d, l=1, seed=seed, dtype=dtype, device=device)
P_beras = SphericalDirections(d=d, l=d, seed=seed, dtype=dtype, device=device)

h_mode = 'constant'
nest = SSZD(P=P_nest, alpha=StepSize(init_value= 7e-4 , mode='sqrt'), h=StepSize(init_value=1e-8,    mode=h_mode))
flax = SSZD(P=P_flax, alpha=StepSize(init_value= 15e-3 , mode='sqrt'), h=StepSize(init_value=1e-8,    mode=h_mode))
coo_d = SSZD(P = P1_c, alpha=StepSize(init_value= 7e-3  , mode='sqrt'), h=StepSize(init_value=1e-8,  mode=h_mode))
sph_d = SSZD(P = P1_s, alpha=StepSize(init_value= 5e-3 , mode='sqrt'), h=StepSize(init_value=1e-8,   mode=h_mode))
duchi = SSZD(P=P_duchi, alpha=StepSize(init_value= 2e-4 , mode='sqrt'), h=StepSize(init_value=1e-8,  mode=h_mode))
berash = SSZD(P=P_beras, alpha=StepSize(init_value= 45e-4 , mode='sqrt'), h=StepSize(init_value=1e-8, mode=h_mode))

init_x = 2

out = "./results/log_reg"

# SSZD
results_nest = run_sszd_experiment(target, init_x, nest, T=T, reps = reps)
store_result(results_nest, "{}/nesterov".format(out))

results_flax = run_sszd_experiment(target, init_x, flax, T=T, reps = reps)
store_result(results_flax, "{}/flax".format(out))

results_duchi = run_sszd_experiment(target, init_x, duchi, T=T, reps = reps)
store_result(results_duchi, "{}/duchi".format(out))

results_berash = run_sszd_experiment(target, init_x, berash, T=T, reps = reps)
store_result(results_berash, "{}/berash".format(out))

#results_sszd_coo = run_sszd_experiment(target, init_x, coo_d, T=T, reps = reps)
#store_result(results_sszd_coo, "{}/sszd_coo_{}".format(out, d))
#
results_sszd_sph = run_sszd_experiment(target, init_x, sph_d, T=T, reps = reps)
store_result(results_sszd_sph, "{}/sszd_sph_{}".format(out, d))
