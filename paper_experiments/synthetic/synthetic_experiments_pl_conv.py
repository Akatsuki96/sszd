import sys
import torch
import time
import numpy as np

from sszd import SSZD
from sszd.stepsize import StepSize
from sszd.direction_matrices import RandomCoordinate, StructuredSphericalDirections

sys.path.append("../")
sys.path.append("../other_methods")
from benchmark_functions import StronglyConvex, PLConvexFunction
from utils import *
from other_methods import *


def run_probds_experiment(name, target, init_x, init_alpha, options, d, T, reps):
    results = []    
    ctime = []
    seed = 121212

    stoc_gen = torch.Generator()
    stoc_gen.manual_seed(seed)
    optimizer = GDS(init_alpha=init_alpha, options=options) 
    for r in range(reps):

        x = torch.full((1, d), init_x, dtype=dtype).to(device)
        y_full = target(x).item()
        results.append([y_full])
        ctime.append([0])
        
        t = 0
        while t < T:
#        for t in range(T):
            it_time = time.time()
            z = torch.randint(low=0, high=d,size=(1,),generator=stoc_gen).item()
            x, num_eval = optimizer.stochastic_step(target, x, z)
            y_full = target(x).item()
            it_time = time.time() - it_time
            print("[{}] reps: {}/{}\tt: {}/{}\ty: {}".format(name, r, reps, t, T, y_full))
            if t + num_eval >= T:
                num_eval = T - t
            ctime[r].append(it_time)
            results[r].append(y_full)
            if num_eval -1 > 0:
                for _ in range(num_eval - 1):
                    results[r].append(y_full)
                    ctime[r].append(0.0)
            t+=num_eval
        optimizer.reset()
    ctime = np.cumsum(ctime, 1)
    
    return (np.mean(results, axis=0), np.std(results, axis=0), np.mean(ctime, axis=0), np.std(ctime, axis=0))
    



def run_stp_experiment(target, init_x, init_alpha, d, T, reps):
    results = []    
    ctime = []
    seed = 121212

    stoc_gen = torch.Generator()
    stoc_gen.manual_seed(seed)
    optimizer = STP(init_alpha=init_alpha, options=GDSOptions(d, gen_strat="random_unit"))
    for r in range(reps):

        x = torch.full((1, d), init_x, dtype=dtype).to(device)
        y_full = target(x).item()
        results.append([y_full])
        ctime.append([0])
        
        for t in range(T):
            it_time = time.time()
            z = torch.randint(low=0, high=d,size=(1,),generator=stoc_gen).item()
            x, _, _ = optimizer.stochastic_step(target, x, z)
            y_full = target(x).item()
            it_time = time.time() - it_time
            print("[STP] reps: {}/{}\tt: {}/{}\ty: {}".format(r, reps, t, T, y_full))
            results[r].append(y_full)
            ctime[r].append(it_time)
        optimizer.reset()
    ctime = np.cumsum(ctime, 1)
    return (np.mean(results, axis=0), np.std(results, axis=0), np.mean(ctime, axis=0), np.std(ctime, axis=0))


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
            z = torch.randint(low=0, high=d,size=(1,),generator=stoc_gen).item()
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




d = 100
seed=1212
dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

target = PLConvexFunction(d=d, dtype=dtype, device=device)

T = 5000
reps = 3

P1_c = RandomCoordinate(d=d, l=d, seed=seed, dtype=dtype, device=device)
P2_c = RandomCoordinate(d=d, l=d//2, seed=seed, dtype=dtype, device=device)
P1_s = StructuredSphericalDirections(d=d, l=d, seed=seed, dtype=dtype, device=device)
P2_s = StructuredSphericalDirections(d=d, l=d//2, seed=seed, dtype=dtype, device=device)
coo_d = SSZD(P = P1_c, alpha=StepSize(init_value= 1e-2  , mode='sqrt'), h=StepSize(init_value=1e-9, mode='constant'))
coo_hd = SSZD(P = P2_c, alpha=StepSize(init_value= 1e-2 *0.5 , mode='sqrt'), h=StepSize(init_value=1e-9, mode='constant'))
sph_d = SSZD(P = P1_s, alpha=StepSize(init_value= 1e-2 , mode='sqrt'), h=StepSize(init_value=1e-9, mode='constant'))
sph_hd = SSZD(P = P2_s, alpha=StepSize(init_value= 1e-2 * 0.5 , mode='sqrt'), h=StepSize(init_value=1e-9, mode='constant'))

init_x = 0.1

out = "./results/pl_convex"

# SSZD
results = run_sszd_experiment(target, init_x, coo_d, T=T, reps = reps)
store_result(results, "{}/sszd_coo_{}".format(out, d))
results = run_sszd_experiment(target, init_x, coo_hd, T=T, reps = reps)
store_result(results, "{}/sszd_coo_{}".format(out, d//2))
results = run_sszd_experiment(target, init_x, sph_d, T=T, reps = reps)
store_result(results, "{}/sszd_sph_{}".format(out, d))
results = run_sszd_experiment(target, init_x, sph_hd, T=T, reps = reps)
store_result(results, "{}/sszd_sph_{}".format(out, d//2))

# STP experiment
results = run_stp_experiment(target, init_x, 10.0, d, T, reps)
store_result(results, "{}/stp".format(out))
#
## ProbDS
probds_indep_opt = GDSOptions(d, alpha_max=1e-1, exp_factor=1.2, cont_factor=0.25, gen_strat="random_unit")
results = run_probds_experiment("ProbDS indep", target, init_x, 1e-1, probds_indep_opt, d, T, reps)
store_result(results, "{}/probds_indep".format(out))

probds_orth_opt = GDSOptions(d, alpha_max=1e-2, exp_factor=1.1, cont_factor=0.15, gen_strat="random_orth")
results = run_probds_experiment("ProbDS orth", target, init_x, 5e-3, probds_orth_opt, d, T, reps)
store_result(results, "{}/probds_orth".format(out))

probds_nh_opt = GDSOptions(d, alpha_max=1e-1, exp_factor=1.001, cont_factor=0.15, gen_strat="n_half")
results = run_probds_experiment("ProbDS nhalf", target, init_x, 1e-1, probds_nh_opt, d, T, reps)
store_result(results, "{}/probds_nhalf".format(out))

probds_rd_ind_opt = GDSOptions(d, alpha_max=1e-1, exp_factor=1.2, cont_factor=0.25, gen_strat="random_unit", sketch=("gaussian", d//2))
results = run_probds_experiment("ProbDS-RD indip", target, init_x, 1e-1, probds_rd_ind_opt, d, T, reps)
store_result(results, "{}/probds_rd_indep".format(out))

probds_rd_ind_opt = GDSOptions(d, alpha_max=1e-2, exp_factor=1.001, cont_factor=0.15, gen_strat="random_orth", sketch=("orthogonal", d//2))
results = run_probds_experiment("ProbDS-RD orth", target, init_x, 1e-1, probds_rd_ind_opt, d, T, reps)
store_result(results, "{}/probds_rd_orth".format(out))
