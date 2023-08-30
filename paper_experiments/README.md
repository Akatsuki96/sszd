# Paper experiments

This directory contains the scripts used to perform the experiments of the paper.

## Required Packages
To run these scripts, you need to install some packages i.e. matplotlib and pandas. You can install them using pip
```
pip install -r requirements.txt
```

## Experiments: Number of directions
To reproduce Fig. 1, you have to run the script `changing_l.py` that you can find in the directory `number_of_direction`

## Experiment: Synthetic comparison
To reproduce Fig. 2, you have to run the scripts `synthetic_experiments_*` scripts inside the directory `synthetic` these will produce the results for the strongly convex, PL convex and non-convex setting. 

To plot the results, you have to execute the `plot_results.py` script modifying the following variables:

- `title`: Title of the plot
- `fname_out`: name of the file in output
- `out` : where results of the script are stored (it is `./results/{name}` where name is for instance `pl_convex` for the PL Convex experiment)

To reproduce the convex (non-pl) experiment, you have to run the script `logreg_exp.py` in the `logreg` folder. To make the plot, you have to run `plot_result.py` in `logreg` directory.

## Falkon Experiments

To reproduce Falkon experiments, you have to install Falkon library. You can find the library and its documentation here: https://github.com/FalkonML/falkon

Note that a CPU version of this library can be installed with pip

```
pip install falkon -f https://falkon.dibris.unige.it/torch-2.0.0_cpu.html
```

To reproduce the experiment on California Housing dataset, you have to run the script `falkon_house.py`. 

To reproduce the experiments on HTRU2 and CASP dataset, you have to download the two datasets and put the csv files in the `data` directory. Then, you have to run the scripts `falkon_htru2.py` and `falkon_casp.py`.

To make the plot, modify the variables `OUT, out_1, out_2, out_3, d` in `plot_results.py` setting:

- `OUT` : path of the results e.g. "./results/casp/"
- `out_1` : name of the output file for the plot with training error
- `out_2` : name of the output file for the plot with validation error
- `out_3` : name of the output file for the plot with cumulative time
- `d` : number of input dimensions (it is 9 for California housing and HTRU2 and it is 10 for CASP)

For instance, for california housing results

```
OUT = './results/house'
out_1 = "house_tr_err"
out_2 = "house_vl_err"
out_3 = "house_time"
d = 9
```

Then execute it.

The datasets can be downloaded from the following links:

- HTRU2: https://archive.ics.uci.edu/dataset/372/htru2
- CASP: https://archive.ics.uci.edu/dataset/265/physicochemical+properties+of+protein+tertiary+structure



**Attention:** since the Falkon library changed, results can be different from the one reported in the paper.

## Finite-difference comparison

To reproduce Fig. 4, you have to run the scripts in `fd_comparison` directory and run `plot_results.py` script to make the plots (the structure of this file is the same of the `plot_results.py` in the previous section).