import numpy as np

import matplotlib.pyplot as plt



def plot_results(results, title, y_ax, out_file_name, ylim=None, logscale=False):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.set_title("{}".format(title), fontsize=20)
    for (val_err_mean, val_err_std, label, color) in results:
        ax.plot(range(val_err_mean.shape[0]), val_err_mean, '-', c="{}".format(color), label="{}".format(label), linewidth=3)
        ax.fill_between(range(val_err_mean.shape[0]), val_err_mean - val_err_std, val_err_mean + val_err_std, alpha=0.2, color="{}".format(color))
    ax.set_xlabel("$k$", fontsize=18)
    ax.set_ylabel("{}".format(y_ax), fontsize=18)
    if ylim is not None:
        ax.set_ylim(ylim)
    if logscale:
        ax.set_yscale("log")
    ax.legend()
        
    plt.savefig("{}.pdf".format(out_file_name), bbox_inches="tight")
    plt.close(fig)
