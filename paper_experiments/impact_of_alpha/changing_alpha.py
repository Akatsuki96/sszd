import sys
sys.path.append("../")

import time

import numpy as np
from szo import SZO

import matplotlib

import matplotlib.pyplot as plt
from opt_result import OptResult
from benchmark_functions import SquareNorm, SquareNormPL, NonConvPL

def plot_results(fname, results, l_values):
    cmap = matplotlib.cm.get_cmap('turbo')
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.set_title("Stochastic function values", fontsize=20)
    ax1.set_xlabel("$k$", fontsize=18)
    ax1.set_ylabel("$F(x_k, \\theta_k)$", fontsize=18)
    for i in range(len(l_values)):
        rgba = cmap((i - 0.01)/len(results))

        avg_ctime, std_ctime, avg_fvalues, std_fvalues, avg_Fvalues, std_Fvalues = results[i].get_mean_std()
        
        lcb = avg_Fvalues - std_Fvalues
        lcb[lcb < 0] = 0.0
        
        ax1.plot(range(avg_Fvalues.shape[0]), avg_Fvalues, '-', color=rgba, label="$c = {}$".format(l_values[i]), linewidth=4)
        ax1.fill_between(range(avg_Fvalues.shape[0]), lcb, avg_Fvalues + std_Fvalues, alpha=0.2, color=rgba)

    ax1.legend(loc="best")
    plt.savefig("{}.png".format(fname), bbox_inches="tight")
    plt.close(fig)


def sq_norm_experiment(dirtype, fn, reps=10):
    d = 100
    l = 5
    alpha_values = [1, 2, 3, 5, 10]
    budget = 1000

    sq_norm10 = fn(d)
          

    
    results = [OptResult(budget, reps) for _ in range(len(alpha_values))]
    for j in range(len(alpha_values)):
        alpha = lambda k: (l/d) * (k**(-1/alpha_values[j])) * 1e-2
        h = lambda k : (1/k**2) #if k > 1 else 1e-3
        rnd_state = np.random.RandomState(12) # state for sampling theta
        optimizer = SZO(dirtype, d, l, alpha, h, dtype=np.float64)
        for i in range(reps):
            
            x = rnd_state.random(d)  # initial guess
            theta = rnd_state.randint(d)
            it_time = time.time()
            _ = sq_norm10(x, theta)
            it_time = time.time() - it_time
            results[j].append_result(i, 0, it_time, sq_norm10.complete(x), sq_norm10(x, theta))

            for k in range(1, budget):
                theta = rnd_state.randint(d) # choose a line of matrix A
                it_time = time.time()
                x, grad, _ = optimizer.stochastic_step(sq_norm10, x, theta)
                it_time = time.time() - it_time
                results[j].append_result(i, k, it_time, sq_norm10.complete(x), sq_norm10(x, theta))
            optimizer.reset()
    return results

alpha_values = [1, 2, 3, 5, 10]

results_coord = sq_norm_experiment("coordinate", SquareNorm, reps=10)
plot_results("conv_coord", results_coord, alpha_values)
print("[CONV] Coordinate completed!")
results_sph = sq_norm_experiment("spherical", SquareNorm, reps=10)
plot_results("conv_sph", results_sph, alpha_values)
print("[CONV] Spherical completed!")
