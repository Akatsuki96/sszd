from sszd.direction_matrices import CoordinateDescentStrategy, SphericalStrategy, GaussianStrategy
from sszd.direction_matrices.dir_strat import DirectionStrategy

UNK_DIR_STRAT = "Unknown strategy to build matrix of direction!\n  Actually you can use only 'coordinate', 'spherical' or a custom strategy (i.e. object of a class that extend DirectionStrategy class)"


def str_strategy(dir_build : str, d : int, l : int, dtype, seed : int):
    if dir_build=='coordinate':
        return CoordinateDescentStrategy(d, l=l, dtype=dtype, seed=seed)
    elif dir_build=='spherical':
        return SphericalStrategy(d, l=l,  dtype=dtype, seed=seed)
    elif dir_build=='gaussian':
        return GaussianStrategy(d, l=l,  dtype=dtype, seed=seed)
    raise Exception(UNK_DIR_STRAT)

def get_strategy(dir_build, d, l, dtype, seed):
    if isinstance(dir_build, DirectionStrategy):
        return dir_build(d, l=l, dtype=dtype, seed=seed)
    elif isinstance(dir_build, str):
        return str_strategy(dir_build, d, l, dtype, seed)
    raise Exception(UNK_DIR_STRAT)
