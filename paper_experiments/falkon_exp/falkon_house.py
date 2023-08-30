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


M = 100
Xtr, Xte, ytr, yte = build_dataset()
#print(Xtr.shape)
#exit()
trsf = PositiveTransform(1e-8)

def mse_err(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

def build_falkon(parameters):
    #parameter[sigma, lam]
    config = {
        'kernel' : GaussianKernel(trsf(parameters[:-1])),
        'penalty' : trsf(parameters[-1]),#),
        'M' : M,
        'seed' : 1234,
        'maxiter' : 10,
        'options' : FalkonOptions(keops_active="auto", use_cpu=True, debug=False)
    }
    return Falkon(**config)


def target(config, z = None):
    config = config.reshape(-1)
    model = build_falkon(config)
    if z is None:
        x_tr, x_te, y_tr, y_te =Xtr, Xte, ytr, yte
    else:        
        x_tr, x_te, y_tr, y_te = z    
    model.fit(x_tr, y_tr)
    return mse_err(y_te, model.predict(x_te).cpu())



def run_probds_experiment(name, target, init_x, init_alpha, options, d, T, reps):
    results = []    
    results_val = []
    ctime = []
    seed = 121212

    stoc_gen = torch.Generator()
    stoc_gen.manual_seed(seed)
    optimizer = GDS(init_alpha=init_alpha, options=options) 
    for r in range(reps):
        state = np.random.RandomState(12)
        X_t, X_v, y_t, y_v = train_test_split(Xtr, ytr, train_size=0.7, random_state=state)
        
        x = torch.full((1, d), init_x, dtype=dtype).to(device)
        y_full = target(x, (X_t, X_t, y_t, y_t)).item()
        y_val = target(x, (X_t, X_v, y_t, y_v)).item()
        results.append([y_full])
        results_val.append([y_val])
        ctime.append([0])
        
        t = 0
        while t < T:
#        for t in range(T):
            it_time = time.time()
            X_t, X_v, y_t, y_v = train_test_split(Xtr, ytr, train_size=0.9, random_state=state)

            x, num_eval = optimizer.stochastic_step(target, x, (X_t, X_v, y_t, y_v))
            # x = torch.clamp(x, min=1e-5, max=100.0)
            y_full = target(x, (X_t, X_t, y_t, y_t)).item()
            y_val = target(x, (X_t, X_v, y_t, y_v)).item()
            it_time = time.time() - it_time
            print("[{}] reps: {}/{}\tt: {}/{}\ty: {}\t{}".format(name, r, reps, t, T, y_full, y_val))
           # if t + num_eval >= T:
           #     num_eval = T - t
            ctime[r].append(it_time)
            results[r].append(y_full)
            results_val[r].append(y_val)
            #if num_eval -1 > 0:
            #    for _ in range(num_eval - 1):
            #        results[r].append(y_full)
            #        results_val[r].append(y_val)
            #        ctime[r].append(0.0)
            t+=1#num_eval
        optimizer.reset()
    ctime = np.cumsum(ctime, 1)
    
    return (
        np.mean(results, axis=0), np.std(results, axis=0), 
        np.mean(results_val, axis=0), np.std(results_val, axis=0), 
        np.mean(ctime, axis=0), np.std(ctime, axis=0))


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
        #x[0, -1] = 1e-5
        X_t, X_v, y_t, y_v = train_test_split(Xtr, ytr, train_size=0.7, random_state=state)
        y_full = target(x, (X_t, X_t, y_t, y_t)).item()
        y_val = target(x, (X_t, X_v, y_t, y_v)).item()

        results.append([y_full])
        results_val.append([y_val])
        #x = torch.full((1, d), 1e-7, dtype=dtype)#.to(device)
        #x[-1] = 1e-7
        ctime.append([0.0])
        for t in range(T):
            it_time = time.time()
            X_t, X_v, y_t, y_v = train_test_split(Xtr, ytr, train_size=0.9, random_state=state)
            x = optimizer.step(target, x, (X_t, X_v, y_t, y_v))
            #print(x)
            # x = torch.clamp(x, min=1e-5, max=100.0)
           # print(x)
            y_full = target(x, (X_t, X_t, y_t, y_t)).item()
            y_val = target(x, (X_t, X_v, y_t, y_v)).item()
            
            it_time = time.time() - it_time
            print("[SSZD] reps: {}/{}\tt: {}/{}\ty_tr: {}\ty_vl: {}".format(r, reps, t, T , y_full, y_val))
            ctime[r].append(it_time)
            results[r].append(y_full)
            results_val[r].append(y_val)
#            for _ in range(l-1):
#                results[r].append(y_full)
#                results_val[r].append(y_val)
#                ctime[r].append(0.0)
        optimizer.reset()
    ctime = np.cumsum(ctime, 1)
    return (
        np.mean(results, axis=0), np.std(results, axis=0), 
        np.mean(results_val, axis=0), np.std(results_val, axis=0), 
        np.mean(ctime, axis=0), np.std(ctime, axis=0))


def run_stp_experiment(target, init_x, init_alpha, d, T, reps):
    results, results_val = [], []    
    ctime = []
    seed = 121212

    stoc_gen = torch.Generator()
    stoc_gen.manual_seed(seed)
    optimizer = STP(init_alpha=init_alpha, options=GDSOptions(d, gen_strat="random_unit", device='cpu'))
    for r in range(reps):
        state = np.random.RandomState(12)
        X_t, X_v, y_t, y_v = train_test_split(Xtr, ytr, train_size=0.7, random_state=state)

        x = torch.full((1, d), init_x, dtype=dtype)#.to(device)
        y_full = target(x, (X_t, X_t, y_t, y_t)).item()
        y_val = target(x, (X_t, X_v, y_t, y_v)).item()
        results.append([y_full])
        results_val.append([y_val])
        ctime.append([0])
        
        for t in range(T):
            it_time = time.time()
            X_t, X_v, y_t, y_v = train_test_split(Xtr, ytr, train_size=0.9, random_state=state)

           # z = torch.randint(low=0, high=d,size=(1,),generator=stoc_gen).item()
            x, _, _ = optimizer.stochastic_step(target, x, (X_t, X_v, y_t, y_v))
            # x = torch.clamp(x, min=1e-5, max=100.0)

            y_full = target(x, (X_t, X_t, y_t, y_t)).item()
            y_val = target(x, (X_t, X_v, y_t, y_v)).item()
            it_time = time.time() - it_time
            print("[STP] reps: {}/{}\tt: {}/{}\ty: {}".format(r, reps, t, T, y_full))
            results[r].append(y_full)
            results_val[r].append(y_val)
            ctime[r].append(it_time)
        optimizer.reset()
    ctime = np.cumsum(ctime, 1)
    return (
        np.mean(results, axis=0), np.std(results, axis=0), 
        np.mean(results_val, axis=0), np.std(results_val, axis=0), 
        np.mean(ctime, axis=0), np.std(ctime, axis=0))



seed=121212
dtype = torch.float32
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
init_x = 9.5

params = torch.full((1, Xtr.shape[1] + 1 ), init_x)#.to(device)
#params[0, -1] = 5
print("[--] Testing...")
print("[--] Test config: ", target(params))
#exit()
T = 30
reps = 2

d = Xtr.shape[1] + 1



P1_c = RandomCoordinate(d=d, l=5, seed=seed, dtype=dtype, device=device)
P1_s = StructuredSphericalDirections(d=d, l=5, seed=seed, dtype=dtype, device=device)
coo_d = SSZD(P = P1_c, alpha=StepSize(init_value= 20.0  , mode='sqrt'), h=StepSize(init_value=1e-2, mode='lin'))
sph_d = SSZD(P = P1_s, alpha=StepSize(init_value= 20.0  , mode='sqrt'), h=StepSize(init_value=1e-2, mode='lin'))


out = "./results/house"

# SSZD
results = run_sszd_experiment(target, init_x, coo_d, T=T, reps = reps)
store_result(results, "{}/sszd_coo_{}".format(out, d))
results = run_sszd_experiment(target, init_x, sph_d, T=T, reps = reps)
store_result(results, "{}/sszd_sph_{}".format(out, d))

# STP experiment
results = run_stp_experiment(target, init_x, 1.0, d, T, reps)
store_result(results, "{}/stp".format(out))
#
# ProbDS
probds_indep_opt = GDSOptions(d, alpha_max=5.0, exp_factor=2.0, cont_factor=0.15, gen_strat="random_unit")
results = run_probds_experiment("ProbDS indep", target, init_x, 5.0, probds_indep_opt, d, T, reps)
store_result(results, "{}/probds_indep".format(out))
#
probds_orth_opt = GDSOptions(d, alpha_max=2.0, exp_factor=1.001, cont_factor=0.5, gen_strat="random_orth")
results = run_probds_experiment("ProbDS orth", target, init_x, 5.0, probds_orth_opt, d, T, reps)
store_result(results, "{}/probds_orth".format(out))
#
probds_nh_opt = GDSOptions(d, alpha_max=1.0, exp_factor=2.0, cont_factor=0.15, gen_strat="n_half")
results = run_probds_experiment("ProbDS nhalf", target, init_x, 1.0, probds_nh_opt, d, T, reps)
store_result(results, "{}/probds_nhalf".format(out))
#
probds_rd_ind_opt = GDSOptions(d, alpha_max=2.0, exp_factor=2.0, cont_factor=0.15, gen_strat="random_unit", sketch=("gaussian", d//2))
results = run_probds_experiment("ProbDS-RD indip", target, init_x, 1.0, probds_rd_ind_opt, d, T, reps)
store_result(results, "{}/probds_rd_indep".format(out))

probds_rd_ind_opt = GDSOptions(d, alpha_max=2.0, exp_factor=2.0, cont_factor=0.5, gen_strat="random_orth", sketch=("orthogonal", d//2))
results = run_probds_experiment("ProbDS-RD orth", target, init_x, 5.0, probds_rd_ind_opt, d, T, reps)
store_result(results, "{}/probds_rd_orth".format(out))
