import torch
import sys
import numpy as np
from math import sqrt
from sszd.optimizers import SSZD
from sszd.direction_matrices import QRDirections, RandomCoordinate

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sys.path.append("../")

from benchmark_functions import LeastSquares


#@jit(nopython=True, parallel=True)
def run_experiment(target, T, reps):
    results = {}
    for i in range(reps):
        print("[++] Repetition: {}/{}".format(i, reps))
        for l in num_directions:

            alpha = lambda k :  0.9 * (1/L) * (l/d) * (1/((k + 1)**(1/2 + 1e-20)))
            h = lambda k :  1e-5 / sqrt(k + 1)# * (1/sqrt(k + 1)) #if 0.1* (1/(k + 1)**2) > 1e-7 else 1e-7
            T_opt = T // (l + 1) if T % l ==0 else T // (l + 1) + 1
            opt = SSZD(target = target, alpha = alpha, h = h, P = QRDirections(d = d, l = l, dtype=dtype))

            ris = opt.optimize(x0=x0, sample_z=sample_z, T = T_opt, verbose=True, return_trace=True)
            fvalues = []
            for fv in ris['fun_values']:
                fvalues += [fv for _ in range(l + 1)]
            if l not in results:
                results[l] = [fvalues]
            else:
                results[l].append(fvalues)
    mean_ris = {}
    for (k, v) in results.items():
        val = np.array(v).reshape(reps, -1)
        mean_ris[k] = (np.mean(val, axis=0), np.std(val, axis=0))
    return mean_ris


def plot_result(mean_ris):

    fig, ax = plt.subplots()
    for(label, (mu, std)) in mean_ris.items():
        ax.plot(range(len(mu)), mu, '-', label=label, lw=3)
        ax.fill_between(range(len(mu)), abs(mu - std), mu + std, alpha=0.4)
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel("number of function evaluations")
    ax.set_ylabel("$f(x_k)$")
    fig.savefig("./test.pdf", bbox_inches='tight')




d = 50
n = 50
num_directions = [1, 5, 25, 50]#, 100]
labels = ["l = {}".format(x) for x in num_directions]
dtype=torch.float64
gen_sampler = torch.Generator()
gen_sampler.manual_seed(121314)
def sample_z():
    return torch.randint(0, high=n, size=(1,), generator=gen_sampler)

L, mu = 500.0, 1.0

target = LeastSquares(n = n, d = d, L=L, mu=mu, x_star=torch.zeros(d, dtype=dtype), dtype=dtype)
x0 = torch.full((d,), 1.0, dtype=dtype) # initialization



T = 100000
reps = 3
mean_ris = run_experiment(target, T=T, reps=reps)

plot_result(mean_ris)