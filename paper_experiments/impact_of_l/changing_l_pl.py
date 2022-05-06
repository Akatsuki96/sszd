import sys
sys.path.append("../")

import time

import numpy as np
import matplotlib
from szo import SZO

import matplotlib.pyplot as plt
from opt_result import OptResult
from benchmark_functions import SquareNorm, SquareNormPL, NonConvPL


#    l_values = [1, 5, 10, 20, 50, 75, 100]

colors = [
    'lightcoral',  
    'orangered', 
    'peru', 
    'darkorange', 
    'gold',
    'palegoldenrod',
    'ivory'
]

def plot_results(fname, results, l_values):
    
    cmap = matplotlib.cm.get_cmap('plasma')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.set_title("Stochastic function values", fontsize=20)
    ax1.set_xlabel("$k$", fontsize=18)
    ax1.set_ylabel("$F(x_k, \\theta_k)$", fontsize=18)
    #ax1.set_title("Function values", fontsize=20)
    #ax1.set_xlabel("$k$", fontsize=18)
    #ax1.set_ylabel("$F(x_k, \\theta_k)$", fontsize=18)
    ax2.set_title("Cumulative Time", fontsize=20)
    ax2.set_xlabel("$k$", fontsize=18)
    ax2.set_ylabel("seconds", fontsize=18)
    for i in range(len(l_values)):
        rgba = cmap(i/len(l_values))
        avg_ctime, std_ctime, avg_fvalues, std_fvalues, avg_Fvalues, std_Fvalues = results[i].get_mean_std()
       # ax1.plot(range(avg_fvalues.shape[0]), avg_fvalues, '-', color=colors[i], label="$l = {}$".format(l_values[i]), linewidth=3)
       # ax1.fill_between(range(avg_fvalues.shape[0]), avg_fvalues - std_fvalues, avg_fvalues + std_fvalues, alpha=0.2, color="{}".format(colors[i]))
        std_Fvalues[std_Fvalues < 0] = 0
        l_F = (avg_Fvalues - std_Fvalues)
        l_time = (avg_ctime - std_ctime)
        l_F[l_F < 0.0] = 0.0
        l_time[l_time < 0.0] = 0.0
        ax1.plot(range(avg_Fvalues.shape[0]), avg_Fvalues, '-', color=rgba, label="$l = {}$".format(l_values[i]), linewidth=4)
        ax1.fill_between(range(avg_Fvalues.shape[0]), l_F, avg_Fvalues + std_Fvalues, alpha=0.2, color=rgba)
        ax2.plot(range(avg_ctime.shape[0]), avg_ctime, '-', color=rgba, label="$l = {}$".format(l_values[i]), linewidth=4)
        ax2.fill_between(range(avg_ctime.shape[0]), l_time, avg_ctime + std_ctime, alpha=0.2, color=rgba)
    ax1.legend(loc="best")
    #ax1.set_yscale("log")
    ax2.set_yscale("log")
    
    ax2.legend(loc="best")
    plt.savefig("{}.png".format(fname), bbox_inches="tight")
    plt.close(fig)


def sq_norm_experiment(dirtype, fn, reps=10):
    d = 100
    l_values = [1, 5, 10, 20, 50, 75, 100]
    budget = 300

    sq_norm10 = fn(d)
          

    
    results = [OptResult(budget, reps) for _ in range(len(l_values))]
    for j in range(len(l_values)):
        rnd_state = np.random.RandomState(12) # state for sampling theta
        alpha = lambda k: float(l_values[j]/d) * (k**(-1/7 + 1e-3)) * 1e-2
        h = lambda k : (1/k ** 2) * 1e-2  #if k > 1 else 1e-3
        optimizer = SZO(dirtype, d, l_values[j], alpha, h, dtype=np.float64)
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
                print("[--] l: {}\ti: {}/{}\tk: {}\ty: {}".format(l_values[j], i,reps, k, sq_norm10(x, theta)))
                results[j].append_result(i, k, it_time, sq_norm10.complete(x), sq_norm10(x, theta))
            optimizer.reset()
            print("[--] l: {}\ti: {}/{}".format(l_values[j], i, reps))
    return results

l_values_sqnor10 = [1, 5, 10, 20, 50, 75, 100]

results_coord = sq_norm_experiment("coordinate", NonConvPL, reps=10)
plot_results("pl_coord", results_coord, l_values_sqnor10)
print("[PL] Coordinate completed!")
results_sph = sq_norm_experiment("spherical", NonConvPL, reps=10)
plot_results("pl_sph", results_sph, l_values_sqnor10)
print("[PL] Spherical completed!")
