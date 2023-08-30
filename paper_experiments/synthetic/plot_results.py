import numpy as np
import matplotlib.pyplot as plt

title = "PL Convex"

out = './results/pl_convex'
fname_out = "pl_convex"

def read_result(label, file):
    mu, std = [], []
    with open("{}/{}/{}.log".format(out, label, file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            splitted = line.split(",")
            mu.append(float(splitted[0]))
            std.append(float(splitted[1]))
    return mu, std

def plot_result(title, labels, file, out, scale=None, legend=True):
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_title("{}".format(title), fontsize=60)
    for (dlab, label) in labels:
    #    if not os.path.exists("./results/logreg_convex/{}/{}.log".format(dlab, file)):
    #        continue
        mu, std = read_result(dlab, file)
        mu, std = np.asarray(mu), np.asarray(std)
        ax.plot(range(len(mu)), mu, '-', lw=8, label=label)
        ax.fill_between(range(len(mu)), mu + std, mu - std, alpha=0.2)
    if legend:
        ax.legend(loc="best", fontsize=38)
    ax.set_xlabel("Function Evaluations", fontsize=50)
    ax.set_ylabel("$f(x_k)$", fontsize=50)
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
    ('sszd_coo_50', 'S-SZD-CO ($l = d/2$)'),
    ('sszd_coo_100', 'S-SZD-CO ($l = d$)'),
    ('sszd_sph_50', 'S-SZD-SP ($l = d/2$)'),
    ('sszd_sph_100', 'S-SZD-SP ($l = d$)')
]
legend = True
plot_result(title, labels, "mean_value", f"{fname_out}_fun_vals", legend=legend)
plot_result(f"{title}: Cumulative Time", labels, "ctime", f"{fname_out}_time", "log", legend=legend)




