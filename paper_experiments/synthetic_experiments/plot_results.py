import numpy as np
import matplotlib.pyplot as plt

import matplotlib

import os

os.environ["XDG_SESSION_TYPE"] = "xcb"

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def read_results(fname):
    mu, sigma = [], []
    with open(fname) as f:
        for line in f.readlines():
            splitted = line.split(",")
            mu.append(float(splitted[0]))
            sigma.append(float(splitted[1]))
    return mu, sigma

d = 50

dir_names = ['sconv', 'plconv', 'plnonconv']

out_files = [
    ('STP', 'other', 'stp.log'),
    ('COBYLA', 'other', 'cobyla.log'),
    ('FD Spherical [l = 1]' , 'fd', 'fd_sp_1.log'),
    ('FD Spherical [l = 50]', 'fd', 'fd_sp_50.log'),
    ('FD Gaussian [l = 1]' , 'fd', 'fd_gauss_1.log'),
    ('FD Gaussian [l = 50]', 'fd', 'fd_gauss_50.log'),
    ('FD Coordinates [l = 1]' , 'fd', 'fd_co_1.log'),
    ('FD Coordinates [l = 50]', 'fd', 'fd_co_50.log'),
    ('ProbDS Independent' , 'ds', 'probds_sp_2.log'),
    ('ProbDS Orthogonal', 'ds', 'probds_ortho_2.log'),
    ('ProbDS-RD Independent' , 'ds', 'probds_sketch_ortho_sp_25_2.log'),
    ('ProbDS-RD Orthogonal', 'ds', 'probds_sketch_ortho_ortho_25_2.log'),
    ('SSZD [l = 25]', 'all', 'sszd_qr_25.log'),
    ('SSZD [l = 50]', 'all', 'sszd_qr_50.log'),
]

mlabels = []
for (name, _, _) in out_files:
    mlabels.append(name)

colors = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:olive',
    'tab:purple',
    'dodgerblue',
    'slategray',
    'crimson',
    'navy',
    'saddlebrown',
    'goldenrod',
    'darkslategray',
    'black',
    'dimgray'

]

out_labels = ['all', 'fd', 'ds', 'other']
titles = ['FD Comparison', 'DS Comparison', 'Other Methods Comparison']
results = {}#olabel : {} for olabel in out_labels if olabel != 'all'}

for d_name in dir_names:    
    if d_name not in results:
        results[d_name] = {}
    for (label, plabel, fl) in out_files:
        if '{}_{}'.format(label,plabel) not in results[d_name]:
            results[d_name]['{}_{}'.format(label,plabel)] = read_results(f"{d_name}/{d}/{fl}")


figsize = (16, 5)
fig, axs = plt.subplots(1, 3, figsize=figsize)

ax0 = axs
i = 0
# for i in range(len(axs)):
#     print(titles[i])
#     ax0 = axs[i]
for j in range(len(dir_names)):
    dlabel = dir_names[j]
    for color,mlab in zip(colors,mlabels):
#            print(mlab)
        l1, l2 = f'{mlab}_all', f'{mlab}_{out_labels[i+1]}'
        if l1 in results[dlabel] or l2 in results[dlabel]:
            label = l1 if l1 in results[dlabel] else l2
            mu, std = results[dlabel][label]
            mu, std = np.array(mu), np.array(std)
#                print(mlab, color)
            ax0[j].plot(range(len(mu)), mu, '-', lw=3, c=color, label=label.split("_")[0])
            ax0[j].fill_between(range(len(mu)), abs(mu - std), mu + std, color=color, alpha=0.25)
    ax0[j].set_yscale('log')
    ax0[j].set_ylabel("$f(x_k)$", fontsize=14)
    ax0[j].set_xlabel("function evaluations", fontsize=14)

fig.suptitle("Comparison with Finite-difference Methods", fontsize=20)
print("-"*50)
axs[0].legend(loc='best')
# axs[1, 0].legend(loc='best')
# axs[2, 0].legend(loc='best')

axs[0].set_title("Strongly Convex", fontsize=16)
axs[1].set_title("PL Convex", fontsize=16)
axs[2].set_title("PL Non-Convex", fontsize=16)

axs[0].grid()
axs[1].grid()
axs[2].grid()

fig.tight_layout()

fig.savefig("./fd_comparison.png", bbox_inches='tight')
plt.close(fig)


fig, axs = plt.subplots(1, 3, figsize=figsize)

ax0 = axs
i = 1
# for i in range(len(axs)):
#     print(titles[i])
#     ax0 = axs[i]
for j in range(len(dir_names)):
    dlabel = dir_names[j]
    for color,mlab in zip(colors,mlabels):
#            print(mlab)
        l1, l2 = f'{mlab}_all', f'{mlab}_{out_labels[i+1]}'
        if l1 in results[dlabel] or l2 in results[dlabel]:
            label = l1 if l1 in results[dlabel] else l2
            mu, std = results[dlabel][label]
            mu, std = np.array(mu), np.array(std)
#                print(mlab, color)
            ax0[j].plot(range(len(mu)), mu, '-', lw=3, c=color, label=label.split("_")[0])
            ax0[j].fill_between(range(len(mu)), abs(mu - std), mu + std, color=color, alpha=0.25)
    ax0[j].set_yscale('log')
    ax0[j].set_ylabel("$f(x_k)$", fontsize=14)
    ax0[j].set_xlabel("function evaluations", fontsize=14)

fig.suptitle("Comparison with Direct-search Methods", fontsize=20)
print("-"*50)
axs[0].legend(loc='best')
# axs[1, 0].legend(loc='best')
# axs[2, 0].legend(loc='best')

axs[0].set_title("Strongly Convex", fontsize=16)
axs[1].set_title("PL Convex", fontsize=16)
axs[2].set_title("PL Non-Convex", fontsize=16)

axs[0].grid()
axs[1].grid()
axs[2].grid()

fig.tight_layout()

fig.savefig("./ds_comparison.png", bbox_inches='tight')
plt.close(fig)


fig, axs = plt.subplots(1, 3, figsize=figsize)

ax0 = axs
i = 2
# for i in range(len(axs)):
#     print(titles[i])
#     ax0 = axs[i]
for j in range(len(dir_names)):
    dlabel = dir_names[j]
    for color,mlab in zip(colors,mlabels):
#            print(mlab)
        l1, l2 = f'{mlab}_all', f'{mlab}_{out_labels[i+1]}'
        if l1 in results[dlabel] or l2 in results[dlabel]:
            label = l1 if l1 in results[dlabel] else l2
            mu, std = results[dlabel][label]
            mu, std = np.array(mu), np.array(std)
#                print(mlab, color)
            ax0[j].plot(range(len(mu)), mu, '-', lw=3, c=color, label=label.split("_")[0])
            ax0[j].fill_between(range(len(mu)), abs(mu - std), mu + std, color=color, alpha=0.25)
    ax0[j].set_yscale('log')
    ax0[j].set_ylabel("$f(x_k)$", fontsize=14)
    ax0[j].set_xlabel("function evaluations", fontsize=14)

fig.suptitle("Comparison with Other Methods", fontsize=20)
print("-"*50)
axs[0].legend(loc='best')
# axs[1, 0].legend(loc='best')
# axs[2, 0].legend(loc='best')

axs[0].set_title("Strongly Convex", fontsize=16)
axs[1].set_title("PL Convex", fontsize=16)
axs[2].set_title("PL Non-Convex", fontsize=16)
axs[0].grid()
axs[1].grid()
axs[2].grid()

fig.tight_layout()

fig.savefig("./other_comparison.png", bbox_inches='tight')
plt.close(fig)


