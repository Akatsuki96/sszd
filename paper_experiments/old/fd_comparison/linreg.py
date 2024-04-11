import sys

import matplotlib.pyplot as plt

sys.path.append("../")
from benchmark_functions import StronglyConvex
import torch
import time
import numpy as np

from utils import store_result

from sszd import SSZD
from sszd.stepsize import StepSize
from sszd.direction_matrices import RandomCoordinate, StructuredSphericalDirections, SphericalDirections, GaussianDirections

def generate_data(N, d, dtype, gen):
    w = torch.ones((1, d), dtype=dtype, device=device)
    X = torch.rand(N, d, dtype=dtype, device=device, generator=gen)
    return  X, X @ w.T

d = 100
seed=121212
dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

N = 1000


data_get = torch.Generator(device=device)
data_get.manual_seed(seed)
X_full, Y_full = generate_data(N, d, dtype, data_get)

def target(w, z = None):
    if z is None:
        X, Y = X_full, Y_full
    else:
        X, Y = z
    return ((X @ w.T) - Y).square().mean()


def run_sszd_experiment(target, init_x, optimizer, T, reps):
    dtype = torch.float64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
coo_d = SSZD(P = P1_c, alpha=StepSize(init_value= 7e-3  , mode='sqrt'), h=StepSize(init_value=1e-8,  mode=h_mode))
sph_d = SSZD(P = P1_s, alpha=StepSize(init_value= 7e-3 , mode='sqrt'), h=StepSize(init_value=1e-8,   mode=h_mode))
nest = SSZD(P=P_nest, alpha=StepSize(init_value= 7e-4 , mode='sqrt'), h=StepSize(init_value=1e-8,    mode=h_mode))
duchi = SSZD(P=P_duchi, alpha=StepSize(init_value= 7e-5 , mode='sqrt'), h=StepSize(init_value=1e-8,  mode=h_mode))
flax = SSZD(P=P_flax, alpha=StepSize(init_value= 5e-2 , mode='sqrt'), h=StepSize(init_value=1e-8,    mode=h_mode))
berash = SSZD(P=P_beras, alpha=StepSize(init_value= 7e-3 , mode='sqrt'), h=StepSize(init_value=1e-8, mode=h_mode))

init_x = 0.01

out = "./results/linear_reg"

# SSZD
results_nest = run_sszd_experiment(target, init_x, nest, T=T, reps = reps)
store_result(results_nest, "{}/nesterov".format(out))

results_duchi = run_sszd_experiment(target, init_x, duchi, T=T, reps = reps)
store_result(results_duchi, "{}/duchi".format(out))

results_flax = run_sszd_experiment(target, init_x, flax, T=T, reps = reps)
store_result(results_flax, "{}/flax".format(out))
#
results_berash = run_sszd_experiment(target, init_x, berash, T=T, reps = reps)
store_result(results_berash, "{}/berash".format(out))
#
#results_sszd_coo = run_sszd_experiment(target, init_x, coo_d, T=T, reps = reps)
#store_result(results_sszd_coo, "{}/sszd_coo_{}".format(out, d))

results_sszd_sph = run_sszd_experiment(target, init_x, sph_d, T=T, reps = reps)
store_result(results_sszd_sph, "{}/sszd_sph_{}".format(out, d))
