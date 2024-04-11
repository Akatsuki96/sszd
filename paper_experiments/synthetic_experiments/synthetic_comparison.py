import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from math import sqrt

from sszd.optimizers import SSZD, STP, RandomSearch, DDS, SketchDS, SVRZ
from sszd.direction_matrices import QRDirections, RandomCoordinate, SphericalDirections, GaussianDirections

sys.path.append("../")
from benchmark_functions import LeastSquares, PLConvex, PLNonConvex

from utils import run_algorithm, write_result, run_scipy_agorithm


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


def get_target_function(fun_name, n, d):
    x_star = torch.zeros(d, dtype=dtype, device=device)
    if fun_name == 'sconv':
        return LeastSquares(n = n, d = d, L = 10.0, mu = 5.0, x_star=x_star, device=device, dtype=dtype, seed=seed)
    elif fun_name == 'plconv':
        return PLConvex(n=n, d=d, L = 10.0, x_star=x_star, device=device, dtype=dtype, seed=seed)
    elif fun_name == 'plnonconv':
        return PLNonConvex(d = d, L = 10.0, mu = 5.0, dtype=dtype, device=device, seed = seed)

def main(args):
    n = d = args.d
    T = args.budget

    generator = torch.Generator()
    generator.manual_seed(seed)

#    x_star = torch.zeros(d, dtype=dtype, device=device)
#    target = LeastSquares(n = n, d = d, L = 10.0, mu = 5.0, x_star=x_star, device=device, dtype=dtype, seed=seed)

    target = get_target_function(args.target, n, d)

    def sample_z():
        return torch.randint(0, n, size=(1, ), generator=generator).item()

    if args.target == 'plnonconv':
        x0 = torch.full((d,), 0.10, dtype=dtype, device=device)
    else:
        x0 = torch.full((d,), 1.0, dtype=dtype, device=device)

    if args.opt_name in ['cobyla']:
        run_scipy_agorithm("./{}/{}".format(args.target, d), args.out_file, args.opt_name, target, x0, sample_z, T, reps=args.reps)
    else:
        opt = get_algorithm(target, d, T, args.opt_name, args)

        run_algorithm("./{}/{}".format(args.target,d), args.out_file, opt, x0, sample_z, T, reps=args.reps)


if __name__ == '__main__':
    dirtypes = ['coordinates', 'orthogonal', 'spherical', 'gaussian']
    methods = ['stp', 'sszd', 'isszd', 'random-search', 'ds', 'sds', 'cobyla', 'zo_svrg_coo', 'zo_svrg_coo_rand']
    parser = argparse.ArgumentParser(description="Synthetic experiment on strongly convex target.")
    parser.add_argument("opt_name", type = str, choices=methods, help="The name of the optimizer")
    parser.add_argument("d", type = int,  help="Dimension of the search space")
    parser.add_argument("budget", type = int,  help="Budget of function evaluations")
    
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
    parser.add_argument('--target', type=str, default='sconv', choices=['sconv', 'plconv', 'conv', 'plnonconv'], help='Target function')
    parser.add_argument('--reps', type=int, default=1, help='Number of repetitions.')
    parser.add_argument('--out-file', type=str, default="test_trace", help='Name of the output file.')

    args = parser.parse_args()
    main(args)




