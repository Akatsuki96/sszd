import numpy as np
import argparse as ap

import matplotlib
import matplotlib.pyplot as plt

plt.rc('font', size=12) 

files = [
    'compass_', 
    'sk_gauss_',
    'sk_orth_',
    'gauss_',
    'orth_',
    'n_half_',
    'stp_',
    'szo_coo_',
    'szo_sph_'
    ]

labels = [
    'CS',
    'ProbDS-RD independent',
    'ProbDS-RD orthogonal',
    'ProbDS independent',
    'ProbDS orthogonal',
    'ProbDS d/2',
    'STP',
    'SZD (coordinate)',
    'SZD (spherical)'
]

colors = [
    'lightskyblue',
    'indigo',
    'cadetblue',
    'purple',
    'palevioletred',
    'mediumturquoise',
    'peru',
    'darkolivegreen',
    'black'
]

def read_te(dataset_name):
    results = []
    for i in range(len(files)):
        fname = files[i] + "te_" + dataset_name + ".log"
        with open(fname, 'r') as f:
            # (avg_ctime[i], std_ctime[i], avg_terr[i], std_terr[i], avg_verr[i], std_verr[i]))
            te = [float(x) for x in f.readlines()]
            avg_terr, std_terr = np.mean(te), np.std(te)
        results.append(
            (avg_terr, std_terr, i)
        )
    return results



def read_results(dataset_name):
    results = []
    for i in range(len(files)):
        avg_ctime, std_ctime = [], []
        avg_tr_err, std_tr_err = [], []
        avg_v_err, std_v_err = [], []
        fname = files[i] + dataset_name + ".log"
        with open(fname, 'r') as f:
            # (avg_ctime[i], std_ctime[i], avg_terr[i], std_terr[i], avg_verr[i], std_verr[i]))
            lines = f.readlines()
        for line in lines:
            splitted = line.split(",")
            avg_ctime.append(float(splitted[0]))
            std_ctime.append(float(splitted[1]))
            avg_tr_err.append(float(splitted[2]))
            std_tr_err.append(float(splitted[3]))
            avg_v_err.append(float(splitted[4]))
            std_v_err.append(float(splitted[5]))
        results.append(
            (avg_ctime, std_ctime, avg_tr_err, std_tr_err, avg_v_err, std_v_err, i)
        )
    return results


def plot_results(dataset_name):
    cmap = matplotlib.cm.get_cmap('turbo')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    ax1.set_title("Training error", fontsize=20)
    ax2.set_title("Validation error", fontsize=20)
    ax3.set_title("Cumulative time", fontsize=20)
    results = read_results(dataset_name)
    for (avg_ctime, std_ctime, avg_tr_err, std_tr_err, avg_v_err, std_v_err, i) in results:
        rgba = cmap((i - 0.01)/len(results))
        ax1.plot(range(len(avg_tr_err)), avg_tr_err, '-', label=labels[i], c=rgba, lw=4)
        ax1.fill_between(range(len(avg_tr_err)), np.array(avg_tr_err) - np.array(std_tr_err), np.array(avg_tr_err) + np.array(std_tr_err),  alpha=0.3, color=rgba)
        ax2.plot(range(len(avg_v_err)), avg_v_err, '-', label=labels[i], c=rgba, lw=4)
        ax2.fill_between(range(len(avg_v_err)), np.array(avg_v_err) - np.array(std_v_err), np.array(avg_v_err) + np.array(std_v_err),  alpha=0.3, color=rgba)
        ax3.plot(range(len(avg_ctime)), avg_ctime, '-', label=labels[i], c=rgba, lw=4)
        ax3.fill_between(range(len(avg_ctime)), np.array(avg_ctime) - np.array(std_ctime), np.array(avg_ctime) + np.array(std_ctime),  alpha=0.3, color=rgba)
    
    ax1.set_xlabel("k", fontsize=18)
    ax2.set_xlabel("k", fontsize=18)
    ax3.set_xlabel("k", fontsize=18)
    ax1.set_ylabel("MSE", fontsize=18)
    ax2.set_ylabel("MSE", fontsize=18)
    ax3.set_ylabel("seconds", fontsize=18)
    ax1.legend()
    ax2.legend()
    ax3.legend()
   # ax1.set_yscale("log")
   # ax2.set_yscale("log")
    ax3.set_yscale("log")
    plt.subplots_adjust(wspace=0.3)
    plt.savefig("{}_tuning.png".format(dataset_name), bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('dataset_name', type=str, help="Dataset name in lower-case")    
    
    args = parser.parse_args()
    plot_results(args.dataset_name)
    te_res = read_te(args.dataset_name)
    print("[--] Dataset: {}".format(args.dataset_name))
    for i in range(len(labels)):
        print("\t[--] lab: {}\tte: ${} \pm {}$".format(labels[i], round(te_res[i][0], 4), round(te_res[i][1], 4)))
    
    
