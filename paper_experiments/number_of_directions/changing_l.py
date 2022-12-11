import sys
import torch
import time 
import numpy as np

sys.path.append("../")

from sszd import SSZD
from sszd.stepsize import StepSize
from sszd.direction_matrices import RandomCoordinate, StructuredSphericalDirections
from benchmark_functions import StronglyConvex, PLNonConvexFunction

import matplotlib.pyplot as plt

def run_experiment(direc_class, target, l_list, init_x = 0.005, h_mode='constant', alpha_mode='sqrt', init_alpha = 4e-3, init_h = 1e-8, T = 100, reps=5):
    dtype = torch.float64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = {}
    for l in l_list:
        results[l] = []
        
    for l in l_list:
        print("[--] Executing for l = {}".format(l))
        seed = 121212

        alpha = StepSize(init_value=  init_alpha * l/d, mode=alpha_mode)
        if h_mode == 'constant':
            h = StepSize(init_value=init_h , mode=h_mode)
        else:
            h = StepSize(init_value=init_h * l/d, mode=h_mode)
        P = direc_class(d=d, l=l, seed=seed, dtype=dtype, device=device)
        optimizer = SSZD(P = P, alpha=alpha, h=h)

        stoc_gen = torch.Generator()
        stoc_gen.manual_seed(seed)

        for r in range(reps):

            x = torch.full((1, d), init_x, dtype=dtype).to(device)
            y_full = target(x).item()
            results[l].append([y_full for _ in range(l)])

            for t in range(T//l - 1):
                z = torch.randint(low=0, high=d,size=(1,),generator=stoc_gen).item()
                x = optimizer.step(target, x, z)
                y_full = target(x).item()
                for _ in range(l):
                    results[l][r].append(y_full)
            optimizer.reset()
        results[l] = (np.mean(results[l], axis=0), np.std(results[l], axis=0))
    return results






def plot_results(title, results, l_list, out):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_title("{}".format(title), fontsize=36)

    for l in l_list:
        ax.plot(range(results[l][0].shape[0]), results[l][0], '-', lw=7, label="$\ell = {}$".format(l))
        ax.fill_between(range(results[l][0].shape[0]), 
                        results[l][0] + results[l][1], 
                        results[l][0] - results[l][1], alpha=0.2)

    ax.legend(loc="lower left", fontsize=28)
    ax.set_xlabel("Function Evaluations", fontsize=32)
    ax.set_ylabel("$f(x_k)$", fontsize=32)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.tick_params(axis='both', which='minor', labelsize=26)
    fig.savefig("./{}.pdf".format(out), bbox_inches='tight', transparent=True)
    fig.savefig("./{}.png".format(out), bbox_inches='tight', transparent=True)
    plt.close(fig)


d = 250
seed = 121212
dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("[--] Experiment: increasing the number of directions")
print("[--] Device: {}".format(device))
sconv_target = StronglyConvex(d, dtype=dtype, device=device)
nconv_target = PLNonConvexFunction(d, dtype=dtype, device=device)

l_list = [1, 10, 50, 100, 200, 250]
reps = 5

T = 5000

results_coord = run_experiment(RandomCoordinate, nconv_target, l_list, init_alpha=5e-3, init_h = 1e-7, h_mode='constant', alpha_mode='lin', init_x=0.01, T = T, reps=reps)
plot_results("Coordinate Directions [Non convex Target]", results_coord, l_list, "pl_coord")

results_sph = run_experiment(StructuredSphericalDirections, nconv_target, l_list, init_alpha=5e-3, init_h = 1e-7, h_mode='constant', alpha_mode='lin', init_x=0.01, T = T, reps=reps) 
plot_results("Spherical Directions [Non convex Target]", results_sph, l_list, "pl_sph")


results_coord = run_experiment(RandomCoordinate, sconv_target, l_list, T = T, reps=reps)
plot_results("Coordinate Directions [Convex Target]", results_coord, l_list, "conv_coord")

results_sph = run_experiment(StructuredSphericalDirections, sconv_target, l_list, T = T, reps=reps) 
plot_results("Spherical Directions [Convex Target]", results_sph, l_list, "conv_sph")

