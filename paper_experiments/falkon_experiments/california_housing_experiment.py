import torch
import sys
import os
import argparse
import numpy as np
import pandas as pd
from math import sqrt


sys.path.append("../")

from utils import read_htru2_data

from sklearn.model_selection import train_test_split
from falkon import Falkon
from falkon.kernels import GaussianKernel
from falkon.center_selection import CenterSelector
from falkon.options import FalkonOptions


from sszd import SSZD, DDS, SketchDS, RandomSearch, STP, SVRZ
from sszd.optimizers.opt import Optimizer
from sszd.direction_matrices import QRDirections, RandomCoordinate, GaussianDirections, SphericalDirections
from falkon.hopt.objectives.transforms import PositiveTransform
from sklearn import datasets

rnd_state = np.random.RandomState(12131415)

X, y = datasets.fetch_california_housing(return_X_y=True)

X = (X - X.mean()) / X.std()

y = ((y - y.min()) / (y.max() - y.min()))

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=rnd_state, shuffle=True)



X_test, Y_test = torch.from_numpy(X_test), torch.from_numpy(Y_test)

X_tr, X_vl, y_tr, y_vl = train_test_split(X_train, Y_train, test_size=0.33, shuffle=True, random_state=rnd_state)

X_train, Y_train = torch.from_numpy(X_train), torch.from_numpy(Y_train)

M = int(np.sqrt(X_tr.shape[0]))

d = X_tr.shape[1] + 1


transform = PositiveTransform(1e-5)

X_tr, X_vl, y_tr, y_vl = torch.from_numpy(X_tr), torch.from_numpy(X_vl), torch.from_numpy(y_tr), torch.from_numpy(y_vl)


print((y_tr == -1).sum(), (y_tr == 1).sum())

def mse(pred : torch.Tensor, y : torch.Tensor):
    return (pred - y).square().mean()


device = 'cpu'
dtype = torch.float64
seed = 12131415

def get_polling_matrix(d, l, polling_type, normalize = True):
    if polling_type == 'coordinates':
        return RandomCoordinate(d = d, l= l, normalize=normalize, dtype=dtype, device=device, seed=seed)
    elif polling_type == 'spherical':
        return SphericalDirections(d = d, l= l, dtype=dtype, device=device, seed=seed)
    elif polling_type == 'gaussian':
        return GaussianDirections(d = d, l= l, dtype=dtype, device=device, seed=seed)
    elif polling_type == 'orthogonal':
        return QRDirections(d = d, l= l, normalize=normalize, dtype=dtype, device=device, seed=seed)
    raise Exception("Unknown polling type!")



def get_algorithm(target, d, T, opt_name, params):
    if opt_name == 'sszd':
        alpha = lambda k : params.sszd_alpha * (params.sszd_l / d) *  (1 / (k + 1)**(1/2 + 1e-10)) #if k < 100 else  (params.sszd_l / d) *  (1 / (k + 1)**(1/2 + 1e-10)) 
        h = lambda k : params.sszd_h * (1 / sqrt(k + 1))
        P = get_polling_matrix(d, params.sszd_l, params.sszd_dirtype, normalize=bool(params.sszd_normalize))
        return  SSZD(target=target, alpha=alpha, h=h, P=P)
    elif opt_name == 'isszd':
        alpha = lambda k : params.sszd_alpha * (params.sszd_l / d) *  (1 / (k + 1)**(1/2 + 1e-10)) if k < params.isszd_m else  params.isszd_alpha * sqrt(params.sszd_l / d) *  (1 / (k + 1)**(1/2 + 1e-10)) 
        h = lambda k : params.sszd_h * (1 / sqrt(k + 1))
        P = get_polling_matrix(d, params.sszd_l, params.sszd_dirtype)
        return SSZD(target=target, alpha=alpha, h=h, P=P)
    elif opt_name == 'stp':
        return STP(target = target, d = d, alpha = lambda k : params.stp_alpha * (1 / sqrt(k + 1)), device=device, dtype=dtype, seed = seed)
    elif opt_name == 'ds':
        P = get_polling_matrix(d, params.ds_l, params.ds_dirtype, normalize=False)
        return DDS(target = target,  alpha_0=params.ds_alpha, ds_constant=params.ds_constant, theta=params.ds_theta, rho = params.ds_rho, alpha_lims=(params.ds_min_alpha,params.ds_max_alpha), P = P)
    elif opt_name == 'sds':
        r = params.sds_r
        P = get_polling_matrix(r, params.ds_l, params.ds_dirtype, normalize=False)
        G = get_polling_matrix(d, r, params.sds_dirtype, normalize=False)
        return SketchDS(target = target, ds_constant=params.ds_constant,  alpha_0=params.ds_alpha, theta=params.ds_theta, rho = params.ds_rho, alpha_lims=(params.ds_min_alpha,params.ds_max_alpha), P = P, G = G)
    elif opt_name == 'zo_svrg_coo':
        n = d
        m = params.zo_svrg_m
        P = RandomCoordinate(d = d, l= d, normalize=False, dtype=dtype, device=device, seed=seed)
        G = RandomCoordinate(d = d, l= d, normalize=False, dtype=dtype, device=device, seed=seed)
        return SVRZ(target, n = n, m = m, alpha = params.zo_svrg_alpha, h = params.zo_svrg_h, P = P, G = G)
    elif opt_name == 'zo_svrg_coo_rand':
        n = d
        m = params.zo_svrg_m
        P = SphericalDirections(d = d, l= params.zo_svrg_l, dtype=dtype, device=device, seed=seed)
        G = RandomCoordinate(d = d, l= d, normalize=False, dtype=dtype, device=device, seed=seed)
        return SVRZ(target, n = n, m = m, alpha = params.zo_svrg_alpha, h = params.zo_svrg_h, P = P, G = G)
    elif opt_name == 'random-search':
        return RandomSearch(target=target, sigma = params.rs_sigma)
    raise Exception("Unknown optimizer!")


def run_experiment(x0, optimizer : Optimizer, T, reps=1, seed = 1231414):
    training_errors = []
    validation_errors = []
    tr_errors = []
    test_errors = []
    num_evals = []
    rnd_state = np.random.RandomState(seed=seed)
    def sample_z():
        return rnd_state.choice(X_vl.shape[0], size=1, replace=False)
    for r in range(reps):
        print(f"[{r + 1}/{reps}] Training with {optimizer.name}")
        training_errors = [] #.append([])
        def target(x : torch.Tensor, z = None):
            # x = [sigma_1, ..., sigma_d, lam]
            lam =  transform(x[-1])
            sigma = transform(x[:-1])
            kernel = GaussianKernel(sigma=sigma)
            model = Falkon(kernel=kernel, 
                           penalty=lam,  
                           M = M,
                           center_selection='uniform',

                           seed = seed, options=FalkonOptions(use_cpu=True)
                           )
            model.fit(X_tr, y_tr)
            if z is None:
                tr_err, vl_err = mse(model.predict(X_tr), y_tr), mse(model.predict(X_vl), y_vl)
                print(f"\t[--] training error = {tr_err}\tvalidation error = {vl_err}")
                training_errors.append(tr_err)
                return vl_err #(model.predict(X_vl) - y_vl).square().mean()
            return mse(model.predict(X_vl[z, :]), y_vl[z]) #(model.predict(X_vl[z, :]) - y_vl[z]).square().mean()
        optimizer.target = target
        opt_ris = optimizer.optimize(x0.clone(), sample_z=sample_z, T = T, verbose=False, return_trace=True)
        tr_err, vl_err = [], []
        print(len(training_errors), opt_ris['num_evals'])
        for i in range(len(opt_ris['num_evals'])):
            tr_err += [training_errors[i] for _ in range(opt_ris['num_evals'][i])]
            vl_err += [opt_ris['fun_values'][i] for _ in range(opt_ris['num_evals'][i])]
        validation_errors.append(vl_err)
        tr_errors.append(tr_err)


#        num_evals.append(opt_ris['num_evals'])
        model = Falkon(kernel=GaussianKernel(sigma=transform(opt_ris['x'][:-1])), 
                       penalty=transform(opt_ris['x'][-1]), 
                       M = M,
                       center_selection='uniform',
                       seed = seed,
                       options=FalkonOptions(use_cpu=True)
                       )
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        test_errors.append((preds - Y_test).square().mean().item())
        print("-"*55)
    return tr_errors, validation_errors, test_errors#, num_evals



OUT_DIR = "./results/california_housing"

os.makedirs(OUT_DIR, exist_ok=True)

# l = d 

# x = torch.full((d  , ), 1.0, dtype=torch.float64)
# seed = 12131415
# alpha = lambda t : 5.0 * (1/np.sqrt(1 + t)) * (l/d)
# h = lambda t : 1e-2 / np.sqrt(t + 1)
# P = QRDirections(d = d , l = l , dtype=torch.float64, seed=121314)
# opt = SSZD(None, alpha = alpha, h= h, P = P)
# T = 500
# reps = 2
# training_errors, validation_errors, test_errors, num_evals = run_experiment(x, optimizer = opt, T = T, reps = reps, seed=seed)
# mu_tr, sigma_tr = np.mean(training_errors, axis=0), np.std(training_errors, axis=0)
# mu_vl, sigma_vl = np.mean(validation_errors, axis=0), np.std(validation_errors, axis=0)
# mu_te, sigma_te = np.mean(test_errors), np.std(test_errors)
# mu_ne, sigma_ne = np.mean(num_evals, axis=0), np.std(num_evals, axis=0)

# with open(f"{OUT_DIR}/{opt.name}_{l}_trace.log", 'w') as f:
#     for i in range(mu_tr.shape[0]):
#         f.write("{},{},{},{},{},{}\n".format(mu_tr[i], sigma_tr[i], mu_vl[i], sigma_vl[i], mu_ne[i], sigma_ne[i]))
#         f.flush()

# with open(f"{OUT_DIR}/{opt.name}_{l}_test_error.log", 'w') as f:
#     f.write("{},{}\n".format(mu_te, sigma_te))
#     f.flush()

def main(args):
    d = X_tr.shape[1] + 1
    T = args.budget

    generator = torch.Generator()
    generator.manual_seed(seed)
    opt = get_algorithm(None, d, T, args.opt_name, args)
    x0 = torch.full((d  , ), 1.0, dtype=torch.float64)
#    x0 = torch.full((d  , ), 5.0, dtype=torch.float64)

    training_errors, validation_errors, test_errors = run_experiment(x0, opt, T, reps=args.reps, seed = 1231414)

    mu_tr, sigma_tr = np.mean(training_errors, axis=0), np.std(training_errors, axis=0)
    mu_vl, sigma_vl = np.mean(validation_errors, axis=0), np.std(validation_errors, axis=0)
    mu_te, sigma_te = np.mean(test_errors), np.std(test_errors)

    with open(f"{OUT_DIR}/{args.out_file}_trace.log", 'w') as f:
        for i in range(mu_tr.shape[0]):
            f.write("{},{},{},{}\n".format(mu_tr[i], sigma_tr[i], mu_vl[i], sigma_vl[i]))
            f.flush()

    with open(f"{OUT_DIR}/{args.out_file}_test_error.log", 'w') as f:
        f.write("{},{}\n".format(mu_te, sigma_te))
        f.flush()


if __name__ == '__main__':
    dirtypes = ['coordinates', 'orthogonal', 'spherical', 'gaussian']
    methods = ['stp', 'sszd', 'isszd', 'random-search', 'ds', 'sds', 'cobyla', 'zo_svrg_coo', 'zo_svrg_coo_rand']
    parser = argparse.ArgumentParser(description="Synthetic experiment on strongly convex target.")
    parser.add_argument("opt_name", type = str, choices=methods, help="The name of the optimizer")
    parser.add_argument("budget", type = int, default=10,  help="Budget of function evaluations")
    
    # STP arguments
    parser.add_argument('--stp-alpha', type=float, default=5e-2, help='Stepsize constant of STP')

    # SSZD arguments
    parser.add_argument('--sszd-alpha', type=float, default=3e-1, help='Stepsize constant of SSZD.')
    parser.add_argument('--sszd-h', type=float, default=1e-7, help='Discretization constant of SSZD.')
    parser.add_argument('--sszd-l', type=int, default=1, help='Number of directions of SSZD.')
    parser.add_argument('--sszd-normalize', type=int, choices=[0, 1], default=1, help='Normalize directions (1 if you want to use SSZD).')
    parser.add_argument('--sszd-dirtype', type=str, default='orthogonal', choices=dirtypes, help='Direction type of SSZD.')

    parser.add_argument('--isszd-m', type=int, default=150, help='Number of iterations after which stepsize changes of ISSZD.')
    parser.add_argument('--isszd-alpha', type=float, default=2.0, help='Stepsize constant to use after the stepsize change of ISSZD.')


    # Random-search arguments
    parser.add_argument('--rs-sigma', type=float, default=1.0, help='Standard deviation of RS gaussian')

    # Direct-search arguments
    parser.add_argument('--ds-theta', type=float, default=0.5, help='Contraction factor of Direct Search')
    parser.add_argument('--ds-rho', type=float, default=2.0, help='Expansion factor of Direct Search')
    parser.add_argument('--ds-alpha', type=float, default=1.0, help='Initial stepsize of Direct Search')
    parser.add_argument('--ds-constant', type=float, default=10.0, help='Scaling factor for forcing function of Direct Search')
    parser.add_argument('--ds-min-alpha', type=float, default=1e-5, help='Minimum stepsize of Direct Search')
    parser.add_argument('--ds-max-alpha', type=float, default=10.0, help='Maximum stepsize of Direct Search')
    parser.add_argument('--ds-l', type=int, default=1, help='Polling set size.')
    parser.add_argument('--ds-dirtype', type=str, default='spherical', choices=dirtypes, help='Direction type of SSZD.')

    # Sketched DS arguments
    parser.add_argument('--sds-r', type=int, default=1, help='subspace dimension.')
    parser.add_argument('--sds-dirtype', type=str, default='spherical', choices=dirtypes, help='Sketching type.')

    # ZO-SVRG arguments
    parser.add_argument('--zo-svrg-alpha', type=float, default=1.0, help='stepsize for ZO-SVRG-Coord and ZO-SVRG-Coord-Rand')
    parser.add_argument('--zo-svrg-h', type=float, default=1e-7, help="discretization parameter for ZO-SVRG-Coord and ZO-SVRG-Coord-Rand")
    parser.add_argument('--zo-svrg-m', type=int, default=10, help="Update frequency for ZO-SVRG-Coord and ZO-SVRG-Coord-Rand")
    parser.add_argument('--zo-svrg-l', type=int, default=1, help="Number of vectors for  ZO-SVRG-Coord-Rand")

    # Experiment arguments
    parser.add_argument('--reps', type=int, default=1, help='Number of repetitions.')
    parser.add_argument('--out-file', type=str, default="test_trace", help='Name of the output file.')

    args = parser.parse_args()
    main(args)



