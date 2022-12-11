import sys
sys.path.append("../")

import torch
import time

import numpy as np
import pandas as pd

from falkon import Falkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.hopt.objectives.transforms import PositiveTransform
from sszd import SSZD
from sszd.stepsize import StepSize
from sszd.direction_matrices import RandomCoordinate, StructuredSphericalDirections
from sklearn.model_selection import train_test_split

sys.path.append("../")
sys.path.append("../other_methods")
from benchmark_functions import StronglyConvex
from utils import *
from other_methods import *


M = 1000
Xtr, Xte, ytr, yte = load_htru2()
#print(Xtr.shape)
#exit()
trsf = PositiveTransform(1e-9)

def mse_err(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

def build_falkon(parameters):
    #parameter[sigma, lam]
    config = {
        'kernel' : GaussianKernel(parameters[:-1]),
        'penalty' : torch.exp(-parameters[-1]),#),
        'M' : M,
        'maxiter' : 1000,
        'options' : FalkonOptions(keops_active="no", use_cpu=False, debug=False)
    }
    return Falkon(**config)


def target(config, z = None):
    config = config.reshape(-1)
    model = build_falkon(config)
    if z is None:
        x_tr, x_te, y_tr, y_te =Xtr, Xte, ytr, yte
    else:        
        x_tr, x_te, y_tr, y_te = z    
    model.fit(Xtr, ytr)
    return mse_err(yte, model.predict(Xte))


def run_sszd_experiment(target, init_x, optimizer, T, reps):
    dtype = torch.float64
    device = 'cpu' #cuda' if torch.cuda.is_available() else 'cpu'
    d, l = optimizer.P.d, optimizer.P.l
    results, results_val = [], [] 
    ctime = []
    seed = 121212

    stoc_gen = torch.Generator()
    stoc_gen.manual_seed(seed)

    for r in range(reps):
        state = np.random.RandomState(12)
        x = torch.full((1, d), init_x, dtype=dtype)#.to(device)
        x[0, -1] = 1
        X_t, X_v, y_t, y_v = train_test_split(Xtr, ytr, train_size=0.7, random_state=state)
        y_full = target(x, (X_t, X_t, y_t, y_t)).item()
        y_val = target(x, (X_t, X_v, y_t, y_v)).item()

        results.append([y_full for _ in range(l)])
        results_val.append([y_val for _ in range(l)])

        ctime.append([0.0 for _ in range(l)])
        for t in range(T//l - 1):
            it_time = time.time()
            X_t, X_v, y_t, y_v = train_test_split(Xtr, ytr, train_size=0.9, random_state=state)
            x = optimizer.step(target, x, (X_t, X_v, y_t, y_v))
            x = torch.clamp(x, min=1e-9)
            #print(x)
            y_full = target(x, (X_t, X_t, y_t, y_t)).item()
            y_val = target(x, (X_t, X_v, y_t, y_v)).item()
            
            it_time = time.time() - it_time
            print("[SSZD] reps: {}/{}\tt: {}/{}\ty_tr: {}\ty_vl: {}".format(r, reps, t, T//l - 1, y_full, y_val))
            ctime[r].append(it_time)
            results[r].append(y_full)
            results_val[r].append(y_val)
            for _ in range(l-1):
                results[r].append(y_full)
                results_val[r].append(y_val)
                ctime[r].append(0.0)
        optimizer.reset()
    ctime = np.cumsum(ctime, 1)
    return (
        np.mean(results, axis=0), np.std(results, axis=0), 
        np.mean(results_val, axis=0), np.std(results_val, axis=0), 
        np.mean(ctime, axis=0), np.std(ctime, axis=0))

seed=121212
dtype = torch.float64
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
init_x = 1.0

params = torch.full((1, Xtr.shape[1] + 1 ), init_x)#.to(device)
params[0, -1] = 1e-9
print("[--] Testing...")
print("[--] Test config: ", target(params))
#exit()
T = 1000
reps = 3

d = Xtr.shape[1] + 1

P1_c = RandomCoordinate(d=d, l=d, seed=seed, dtype=dtype, device=device)
P1_s = StructuredSphericalDirections(d=d, l=d, seed=seed, dtype=dtype, device=device)
coo_d = SSZD(P = P1_c, alpha=StepSize(init_value= 1e-4  , mode='sqrt'), h=StepSize(init_value=1e-5, mode='constant'))
sph_d = SSZD(P = P1_s, alpha=StepSize(init_value= 1e-1 , mode= 'sqrt'), h=StepSize(init_value=1e-5, mode='constant'))


out = "./results/house"

# SSZD
results = run_sszd_experiment(target, init_x, coo_d, T=T, reps = reps)
store_result(results, "{}/sszd_coo_{}".format(out, d))
results = run_sszd_experiment(target, init_x, sph_d, T=T, reps = reps)
store_result(results, "{}/sszd_sph_{}".format(out, d))
