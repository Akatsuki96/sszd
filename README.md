# S-SZD: Structured Stochastic Zero-th order Descent
Implementation of S-SZD algorithm introduced in *Stochastic Zeroth order Descent with Structured Directions*. 


Instructions to reproduce paper experiments are reported in `paper_experiments/README.md`

## Installation
The library can be installed by cloning the repository and using pip

```
pip install .
```
We suggest to create a conda environment, thus, after cloning the repository, open a terminal inside the sszd directory and execute the following commands

```
conda create -n sszd_env python=3.12.2 numpy=1.26.4 cupy
conda activate sszd_env
pip install .
```


## Citation
If you use this library, please cite it as below
~~~
@misc{sszd,
  doi = {10.48550/ARXIV.2206.05124},
  url = {https://arxiv.org/abs/2206.05124},
  author = {Rando, Marco and Molinari, Cesare and Villa, Silvia and Rosasco, Lorenzo},
  keywords = {Optimization and Control (math.OC), Machine Learning (cs.LG), FOS: Mathematics, FOS: Mathematics, FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Stochastic Zeroth order Descent with Structured Directions},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
~~~
