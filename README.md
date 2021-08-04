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
If you have GPU available we recommend installing the jax version with GPU support, as this 
would speed up the execution of the code by an order of magnitude. The last version 
which was used and tested on the provided code is jax 1.68 (with support for cuda 11.1).

## Usage
The notebooks folder contains examples of code usage and scripts to reproduce the figures 
in the paper. The scripts available in the main folder 

```
run_estimate_runtime.py
run_stationary_sims.py
run_switching_fixed_diff_sims.py
run_switching_varying_diff_sims.py
```
allow for command line execution of simulations in stationary and switching bandits. For example, 
running the following command 

```
python run_estimate_runtime.py -n 1000 -k 10 20 40 80
```
will estimate the runtimes for different decision making algorithms on your machine by running 
1000 parallel simulations for different arm number 10, 20, 40 and 80. The other scripts 
are executed in a similar manner (open the file to find the list of possible commands).

For more details on the definition of different bandit environments see the methods section of the [paper](https://arxiv.org/abs/2101.08699).

As running the simulations can takes long time we provide a pre-generated results at 
the OSF page of the project [osf.io/85ek4/](https://osf.io/85ek4/). Create a data folder 
inside the repository, and download the npz files from the data folder hosted on the project page.
Running the notebooks, stationary_bandits_plotting.ipynb and switching_bandits_comparison.ipynb 
will recreate the figures from the paper.  

## Citing
We will update this part with correct citation as soon as the paper is published. In the 
current absence of peer-reviewed article please cite the arXiv preprint if you decide to use 
the code in your work.