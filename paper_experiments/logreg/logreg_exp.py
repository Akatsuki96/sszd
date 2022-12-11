import sys
import torch
import time

import numpy as np
from sszd import SSZD
from sszd.stepsize import StepSize
from sszd.direction_matrices import RandomCoordinate, StructuredSphericalDirections

sys.path.append("../")
sys.path.append("../other_methods")
from utils import *
from other_methods import *
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

X, Y = build_dataset(N=500, d = 20,device=device)

X = (X - X.mean(dim=0))/X.std(dim=0)

rnd_state = np.random.RandomState(12)

inds = rnd_state.choice(X.shape[0], X.shape[0], replace=False)
X = X[inds, :]
Y = Y[inds]


print(X.shape, Y.shape)

def loss(h, y):
    return (-y * torch.log(h) - (1 - y) * torch.log(1 - h)).mean()

def target(w, z = None):
  #  w = torch.clamp(w, min=0.0)
    if z is None:
#        print((X @ w.T).shape, Y.shape)
#        exit()
        return loss(torch.sigmoid(X @ w.T), Y)
#    print(z)
#    exit()
    x_p, y_p = z[0], z[1]
    return loss(torch.sigmoid(x_p @ w.T), y_p)



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
            idx = torch.randint(low=0, high=X.shape[0],size=(1,),generator=stoc_gen).item()
            z = (X[idx, :], Y[idx])
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
            idx = torch.randint(low=0, high=X.shape[0],size=(1,),generator=stoc_gen).item()
            z = (X[idx, :], Y[idx])
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
#    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d, l = optimizer.P.d, optimizer.P.l
    results = [] 
    ctime = []
    seed = 121212

    stoc_gen = torch.Generator()
    stoc_gen.manual_seed(seed)

    for r in range(reps):

        x = torch.full((1, d), init_x, dtype=dtype, device=device)
        y_full = target(x).item()
        results.append([y_full for _ in range(l)])
        ctime.append([0.0 for _ in range(l)])
        for t in range(T//l - 1):
            it_time = time.time()
            idx = torch.randint(low=0, high=X.shape[0],size=(1,),generator=stoc_gen).item()
            z = (X[idx, :], Y[idx])
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

seed = 121314
dtype = torch.float64

d = X.shape[1]
T = 5000
reps = 3

P1_c = RandomCoordinate(d=d, l=d, seed=seed, dtype=dtype, device=device)
P2_c = RandomCoordinate(d=d, l=d//2, seed=seed, dtype=dtype, device=device)
P1_s = StructuredSphericalDirections(d=d, l=d, seed=seed, dtype=dtype, device=device)
P2_s = StructuredSphericalDirections(d=d, l=d//2, seed=seed, dtype=dtype, device=device)
coo_d = SSZD(P = P1_c, alpha=StepSize(init_value= 1e-2  , mode='sqrt'), h=StepSize(init_value=1e-8, mode='constant'))
coo_hd = SSZD(P = P2_c, alpha=StepSize(init_value= 5e-3  , mode='sqrt'), h=StepSize(init_value=1e-8, mode='constant'))
sph_d = SSZD(P = P1_s, alpha=StepSize(init_value= 1e-2 , mode='sqrt'), h=StepSize(init_value=1e-8, mode='constant'))
sph_hd = SSZD(P = P2_s, alpha=StepSize(init_value= 5e-3 , mode='sqrt'), h=StepSize(init_value=1e-8, mode='constant'))

init_x = 0.2 #torch.zeros(d, dtype=dtype, device=device)

out = "./results/logreg_convex"

# SSZD
#results = run_sszd_experiment(target, init_x, coo_d, T=T, reps = reps)
#store_result(results, "{}/sszd_coo_{}".format(out, d))
#results = run_sszd_experiment(target, init_x, coo_hd, T=T, reps = reps)
#store_result(results, "{}/sszd_coo_{}".format(out, d//2))
#results = run_sszd_experiment(target, init_x, sph_d, T=T, reps = reps)
#store_result(results, "{}/sszd_sph_{}".format(out, d))
#results = run_sszd_experiment(target, init_x, sph_hd, T=T, reps = reps)
#store_result(results, "{}/sszd_sph_{}".format(out, d//2))


# STP experiment
#results = run_stp_experiment(target, init_x, 1.0, d, T, reps)
#store_result(results, "{}/stp".format(out))

# ProbDS
probds_indep_opt = GDSOptions(d, alpha_max=1e-2, exp_factor=1.70, cont_factor=0.5, gen_strat="random_unit")
results = run_probds_experiment("ProbDS indep", target, init_x, 1e-2, probds_indep_opt, d, T, reps)
store_result(results, "{}/probds_indep".format(out))
#
#probds_orth_opt = GDSOptions(d, alpha_max=5e-3, exp_factor=1.0001, cont_factor=0.75, gen_strat="random_orth")
#results = run_probds_experiment("ProbDS orth", target, init_x, 1e-3, probds_orth_opt, d, T, reps)
#store_result(results, "{}/probds_orth".format(out))
#
#probds_nh_opt = GDSOptions(d, alpha_max=1.0, exp_factor=1.000001, cont_factor=0.9, gen_strat="n_half")
#results = run_probds_experiment("ProbDS nhalf", target, init_x, 5e-3, probds_nh_opt, d, T, reps)
#store_result(results, "{}/probds_nhalf".format(out))
#
#probds_rd_ind_opt = GDSOptions(d, alpha_max=5e-3, exp_factor=1.001, cont_factor=0.75, gen_strat="random_unit", sketch=("gaussian", d//2))
#results = run_probds_experiment("ProbDS-RD indip", target, init_x, 1e-3, probds_rd_ind_opt, d, T, reps)
#store_result(results, "{}/probds_rd_indep".format(out))
#
#probds_rd_ind_opt = GDSOptions(d, alpha_max=1e-2, exp_factor=1.001, cont_factor=0.5, gen_strat="random_orth", sketch=("orthogonal", d//2))
#results = run_probds_experiment("ProbDS-RD orth", target, init_x, 1e-3, probds_rd_ind_opt, d, T, reps)
#store_result(results, "{}/probds_rd_orth".format(out))
#