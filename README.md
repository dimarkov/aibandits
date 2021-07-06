# Active inference and multi-armed bandits
The code accompanying the paper [An empirical evaluation of active inference in multi-armed bandits](https://arxiv.org/abs/2101.08699). We introduce various active inference based multi-armed bandit algorithms and compare them to existing solutions. The current focus is on stationary and switching bandits. 

## Installation

The following instruction assume that you use anaconda or miniconda package manager. 
First create the environment using the provided yml file

```
conda env create -f environment.yml
```

Activate the environment

```
conda activate bandits
```
and follow the official instructions for installing [jax](https://github.com/google/jax).
If you have GPU availible we recomend installing the jax version with GPU support, as this 
would speed up the execution of the code by an order of the magnitute. The last version 
which was used and tested on the provided code is jax 1.68 (with support for cuda 11.1).

## Usage
The notebooks folder contains examples of code usage and scripts to reproduce the figures 
in the paper. The scripts availible in the main folder 

```
run_estimate_runtime.py
run_stationary_sims.py
run_switching_fixed_diff_sims.py
run_switching_varying_diff_sims.py
```
allow for comand line execution of simualtions in stationary and switching bandits. For more 
details on the definition of bandits see the methods section of the paper. 

## Citing
We will update this part with correct citation as soon as the paper is published. In the 
current absence of peer-reviwed article please cite the arXiv preprint if you decide to use 
the code in your work.