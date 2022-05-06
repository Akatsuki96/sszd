# Reproducing results of the paper
In this folder, you can find the scripts used to get the results reported in the paper.

To reproduce results of the paper, you need to install:
- szo library 
- some required packages
- Falkon library

## Required packages
In requirements.txt file you can find the required packages to run experiments and reproduce plots. You can install such packages by using pip command in this folder
```
pip install -r requirements.txt
```
In order to avoid dependences conflicts, a brand new python environment is suggested


## Falkon library
To reproduce Falkon experiments, you need to install Falkon library. 
Such library can be downloaded and installed from the following link

Falkon library: https://github.com/FalkonML/falkon

In its documentation you can find instruction to install it.

**Attention:** for Falkon, torch version must be the same of cuda compiler (nvcc).
