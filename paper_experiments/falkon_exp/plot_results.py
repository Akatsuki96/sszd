import os

import numpy as np
import matplotlib.pyplot as plt

OUT = './results/house'
d = 9

def read_result(label, file):
    mu, std = [], []
    with open("{}/{}/{}.log".format(OUT, label, file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            splitted = line.split(",")
            mu.append(float(splitted[0]))
            std.append(float(splitted[1]))
    return mu, std

def plot_result(title, labels, file, out, scale=None, ylabel='$f(x_k)$', legend=True):
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_title("{}".format(title), fontsize=60)
    for (dlab, label) in labels:
        if not os.path.exists("{}/{}/{}.log".format(OUT, dlab, file)):
            continue
        mu, std = read_result(dlab, file)
        mu, std = np.asarray(mu), np.asarray(std)
        ax.plot(range(len(mu)), mu, '-', lw=8, label=label)
        ax.fill_between(range(len(mu)), mu + std, mu - std, alpha=0.2)
    if legend:
        ax.legend(loc="best", fontsize=38)
    ax.set_xlabel("Function Evaluations", fontsize=50)
    ax.set_ylabel(ylabel, fontsize=50)
    if scale is not None:
        ax.set_yscale(scale)
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.tick_params(axis='both', which='minor', labelsize=40)
    fig.savefig("./{}.pdf".format(out), bbox_inches='tight', transparent=True)
    fig.savefig("./{}.png".format(out), bbox_inches='tight', transparent=True)
    plt.close(fig)
   

labels = [
    ('stp', 'STP'),
    ('probds_indep', 'ProbDS - independent'),
    ('probds_orth', 'ProbDS - orthogonal'),
    ('probds_rd_indep', 'ProbDS-RD - independent'),
    ('probds_rd_orth', 'ProbDS-RD - orthogonal'),
    ('probds_nhalf', 'ProbDS d/2'),
    ('sszd_coo_{}'.format(d), 'S-SZD-CO ($l = d$)'),
    ('sszd_sph_{}'.format(d), 'S-SZD-SP ($l = d$)')
]

out_1 = "house_tr_err"
out_2 = "house_vl_err"
out_3 = "house_time"

plot_result("Training Error", labels,   "tr_err", out_1, ylabel="MSE", legend=False)
plot_result("Validation Error", labels, "vl_err", out_2, ylabel="MSE", legend=True)
plot_result("Cumulative Time", labels, "ctime", out_3, "log", ylabel="seconds", legend=False)




