import numpy as np
import matplotlib.pyplot as plt

out = './results/log_reg'

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
    ax.set_title("{}".format(title), fontsize=64)
    i = 1
    for (dlab, label) in labels:
    #    if not os.path.exists("./results/logreg_convex/{}/{}.log".format(dlab, file)):
    #        continue
        mu, std = read_result(dlab, file)
        mu, std = np.asarray(mu), np.asarray(std)
        ax.plot(range(len(mu)), mu, '-', lw=8, label=label)
        ax.fill_between(range(len(mu)), mu + std, mu - std, alpha=0.4)
    if legend:
        ax.legend(loc="best", fontsize=44)
    ax.set_xlabel("Function Evaluations", fontsize=54)
    ax.set_ylabel("$f(x_k)$", fontsize=54)
    if scale is not None:
        ax.set_yscale(scale)
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.tick_params(axis='both', which='minor', labelsize=40)
    fig.savefig("./{}.pdf".format(out), bbox_inches='tight', transparent=True)
    fig.savefig("./{}.png".format(out), bbox_inches='tight', transparent=True)
    plt.close(fig)
   


labels = [
#    ('sszd_coo_100', 'S-SZD-CO'),
    ('sszd_sph_20', 'S-SZD'),
    ('nesterov', 'Nesterov & Spokoiny'),
    ('duchi', 'Duchi et al.'),
    ('flax', 'Flaxman et al.'),
    ('berash', 'Berahas et al.')
]
legend = True
plot_result("Logistic Regression", labels, "mean_value", "logreg_fun_vals", "log", legend=legend)
plot_result("Cumulative Time", labels, "ctime", "logreg_convex_time", "log", legend=legend)




