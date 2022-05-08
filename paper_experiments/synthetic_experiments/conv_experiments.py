import numpy as np
from szo import SZO
import sys

sys.path.append("../")

import time
import matplotlib
import matplotlib.pyplot as plt

from benchmark_functions import SquareNorm, SquareNormPL
from opt_result import OptResult
from other_methods import GDSOptions, GDS, STP

T = 300
max_init, min_init = 2, 1

labels = [
   # 'CS',
    'STP',
    'ProbDS orthogonal',
    'ProbDS-RD orthogonal (l = d/2)',
    'ProbDS independent',
    'ProbDS-RD independent (l = d/2)',
    'ProbDS d/2',
    'SZD-CO (l = d/2)',
    'SZD-CO (l = d)',
    'SZD-SP (l = d)',
    'SZD-SP (l = d/2)',
]
figsize = (10, 4)


def plot_results(results):
    cmap = matplotlib.cm.get_cmap('turbo')

    fig, ax1 = plt.subplots(figsize=figsize)
   # ax1.set_title("Function values")
    ax1.set_title("Stochastic function values: Strongly Convex Function", fontsize=20)
    for i in range(len(results)):
        rgba = cmap((i - 0.01)/len(results))
        avg_ctime, std_ctime, avg_fvalues, std_fvalues, avg_Fvalues, std_Fvalues = results[i].get_mean_std()

        lcb =  avg_Fvalues.reshape(-1) - std_Fvalues.reshape(-1)
        lcb[lcb < 0] = 0.0
        
        ax1.plot(range(avg_Fvalues.shape[0]), avg_Fvalues, c = rgba, label=labels[i], lw=5)
        ax1.fill_between(range(avg_Fvalues.shape[0]),lcb, avg_Fvalues.reshape(-1) + std_Fvalues.reshape(-1),  alpha=0.3, color = rgba)
       
    ax1.set_xlabel("$k$", fontsize=18)
    ax1.set_ylabel("$F(x_k, \\theta_k)$", fontsize=18)

    ax1.legend()
    plt.savefig("conv_vals.png", bbox_inches="tight")
    plt.close(fig)
    fig, ax2 = plt.subplots(figsize=figsize)
    for i in range(len(results)):
        rgba = cmap((i - 0.01)/len(results))

        avg_ctime, std_ctime, avg_fvalues, std_fvalues, avg_Fvalues, std_Fvalues = results[i].get_mean_std()
        
        ax2.plot(range(avg_ctime.shape[0]), avg_ctime, '-', c = rgba, label=labels[i], lw=5)
        ax2.fill_between(range(avg_ctime.shape[0]), avg_ctime - std_ctime, avg_ctime + std_ctime, alpha=0.3, color = rgba)


    ax2.set_title("Cumulative time", fontsize=20)
    ax2.set_xlabel("$k$", fontsize=18)
    ax2.set_ylabel("seconds", fontsize=18)
    
    ax2.set_yscale("log")
    ax2.legend()
    plt.savefig("conv_time.png", bbox_inches="tight")
    plt.close(fig)
def stp_experiment(target, init_alpha, options, reps=5):

    optimizer = STP(init_alpha=init_alpha, options=options)
    result = OptResult(T, reps)
    
    rnd_state = np.random.RandomState(12) # state for sampling theta
    for i in range(reps):
        x = rnd_state.rand(d) *  (max_init - min_init) + min_init # initial guess
        for k in range(T):
            theta = rnd_state.randint(d) # choose a line of matrix A
            it_time = time.time()
            x, grad, _ = optimizer.stochastic_step(target, x, theta)
            it_time = time.time() - it_time
            print("[--] i: {}/{}\tk: {}/{}\t F(x, theta) = {}".format(i, reps, k, T, target(x, theta)))
            result.append_result(i, k, it_time, target.complete(x), target(x, theta))
        optimizer.reset()
    return result

def gds_experiment(target, init_alpha, options, reps=5):

    optimizer = GDS(init_alpha=init_alpha, options=options)
    result = OptResult(T, reps)
    
    rnd_state = np.random.RandomState(12) # state for sampling theta
    for i in range(reps):
        x = rnd_state.rand(d) * (max_init - min_init) + min_init # initial guess
        for k in range(T):
            theta = rnd_state.randint(d) # choose a line of matrix A
            it_time = time.time()
            x, grad = optimizer.stochastic_step(target, x, theta)
            it_time = time.time() - it_time
            print("[--] i: {}/{}\tk: {}/{}\t F(x, theta) = {}".format(i, reps, k, T, target(x, theta)))
            result.append_result(i, k, it_time, target.complete(x), target(x, theta))
        optimizer.reset()
    return result





def szo_experiment(target, dir_type, l, reps=5):

    alpha = lambda k: l/d * (k**(-1/9)) * 1e-4
    h = lambda k : (1/k ** 2) * 1e-4
    optimizer = SZO(dir_type, d, l, alpha, h, dtype=np.float32)
    result = OptResult(T, reps)
    
    rnd_state = np.random.RandomState(12) # state for sampling theta
    for i in range(reps):
        x = rnd_state.rand(d) * (max_init - min_init) + min_init # initial guess
        theta = rnd_state.randint(d)
        it_time = time.time()
        _ = target(x, theta)
        it_time = time.time() - it_time
        
        result.append_result(i, 0, it_time, target.complete(x), target(x, theta))

        for k in range(1, T):
           # theta = rnd_state.randint(d) # choose a line of matrix A
            it_time = time.time()
            x, grad, _ = optimizer.stochastic_step(target, x, theta)
            it_time = time.time() - it_time
            print("[--] i: {}/{}\tk: {}/{}\t F(x, theta) = {}".format(i, reps, k, T, target(x, theta)))

            result.append_result(i, k, it_time, target.complete(x), target(x, theta))
            theta = rnd_state.randint(d)

        optimizer.reset()
    return result

d = 100
l = 100
reps = 5
forc_fun = lambda k: 10*(k**2)

comp_options = GDSOptions(d, alpha_max=5.0, exp_factor=2, cont_factor=0.5)
probds_orth = GDSOptions(d, alpha_max=5.0, exp_factor=2.0, cont_factor=0.5, gen_strat="random_orth")
probds_rd_orth = GDSOptions(d, alpha_max=5.0, exp_factor=2.0, cont_factor=0.5, gen_strat="random_orth", sketch=("orthogonal", d//2))
probds_indep = GDSOptions(d, alpha_max=5.0, exp_factor=2.0, cont_factor=0.5, gen_strat="random_unit")
probds_rd_indep = GDSOptions(d, alpha_max=5.0, exp_factor=2.0, cont_factor=0.5, gen_strat="random_unit", sketch=("gaussian", d//2))
nhalf_indep = GDSOptions(d, alpha_max=5.0, exp_factor=2.0, cont_factor=0.5, gen_strat="n_half")

stp = GDSOptions(d, gen_strat="random_unit")




target = SquareNorm(d)


szo_sph_h = szo_experiment(target, "spherical", l // 2, reps=reps)
szo_coo_h = szo_experiment(target, "coordinate", l // 2, reps=reps)

szo_sph = szo_experiment(target, "spherical", l, reps=reps)
szo_coo = szo_experiment(target, "coordinate", l, reps=reps)
#comp = gds_experiment(target,0.1, comp_options, reps=reps)
probds_orth = gds_experiment(target, 1.0, probds_orth, reps=reps)
probds_rd_orth = gds_experiment(target, 1.0, probds_rd_orth, reps=reps)
probds_indep = gds_experiment(target, 1.0, probds_indep, reps=reps)
probds_rd_indep = gds_experiment(target, 1.0, probds_rd_indep, reps=reps)
nhalf_indep = gds_experiment(target, 1.0, nhalf_indep, reps=reps)
stp = stp_experiment(target, 1.0, stp, reps=reps)

plot_results([
    #comp,
    stp,
    probds_orth,
    probds_rd_orth,
    probds_indep,
    probds_rd_indep,
    nhalf_indep,
    szo_coo_h,
    szo_coo, 
    szo_sph,
    szo_sph_h
])