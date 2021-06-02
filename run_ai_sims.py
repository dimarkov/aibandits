from jax._src.random import PRNGKey
import jax.numpy as jnp
import numpy as np
import itertools
import argparse
import os
import glob

from collections import defaultdict
from tqdm import trange, tqdm
from jax import random, lax, vmap

from bandits import efe_selection, app_selection, ai_sampling_selection, thompson_selection, ots_selection, ucb_selection, bucb_selection
from bandits import learning_stationary as learning
from bandits import generative_process_swtch
from bandits import simulator

from jax import devices
devices(backend='gpu')

rho = .0
log_pj_j = jnp.log(jnp.array([[1 - rho, rho], [1., 0.]]))
process = lambda *args: generative_process_swtch(*args, log_pj_j)

def merge_files(args):
    merged_res = {}
    for K in args.num_arms:
        tmp = np.load('data/tmp_res_K{}_e{}.npz'.format(K, args.difficulty), allow_pickle=True)
        for key in tmp.keys():
            merged_res[key] = tmp[key]
    np.savez('data/stationary_Ks_e{}'.format(args.difficulty), **merged_res)
    
    # delete tmp files
    files = glob.glob(os.path.join('data/tmp_*_e{}.npz'.format(args.difficulty)))
    for file in files:
        os.remove(file)

def main(args):
    N = args.num_runs
    p = args.trial_power
    Ks = args.num_arms
    eps = args.difficulty/100.

    steps = (p - 2) * 9

    lambdas = jnp.arange(.0, 1.5, .015)

    seed = PRNGKey(10396145)

    for K in tqdm(Ks):
        regret_all = defaultdict()
        for func, label in zip([efe_selection, app_selection, ai_sampling_selection], ['EFE_K{}'.format(K), 'APP_K{}'.format(K), 'SMP_K{}'.format(K),]):
            seed, _seed = random.split(seed)

            sim = lambda l: simulator(process, learning, lambda *args: func(*args, lam=l), N=N, steps=steps, K=K, eps=eps, seed=_seed[0])
            results = vmap(sim)(lambdas)
            cum_regret = np.array(results).astype(np.float32)
            regret_all[label] = cum_regret
        
        for func, label in zip([thompson_selection, ots_selection, ucb_selection, bucb_selection], 
                       ['TS_K{}'.format(K), 'OTS_K{}'.format(K), 'UCB_K{}'.format(K), 'BUCB_K{}'.format(K)]):

            seed, _seed = random.split(seed)

            cum_regret = simulator(process, learning, func, N=N, steps=steps, K=K, eps=eps, seed=_seed[0])
            cum_regret = np.array(results).astype(np.float32)
            regret_all[label] = cum_regret
        
        np.savez('data/tmp_res_K{}_e{}'.format(K, args.difficulty), **regret_all)
    
    merge_files(args)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="All bandit algos in classical multi-armed bandits")
    parser.add_argument("-p", "--trial-power", nargs="?", default=3, type=int) # number of trials T = 10^p
    parser.add_argument("-n", "--num-runs", nargs='?', default=10, type=int)
    parser.add_argument("-k", "--num-arms", nargs='+', default=5, type=int)
    parser.add_argument("-d", "--difficulty", nargs='?', default=25, type=int)

    args = parser.parse_args()

    main(args)
