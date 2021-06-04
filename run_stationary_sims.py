import jax.numpy as jnp
import numpy as np
import itertools
import argparse
import os
import glob
import jax

from time import time

from collections import defaultdict
from tqdm import trange, tqdm
from jax import random, lax, vmap
from jax._src.random import PRNGKey

from bandits import efe_selection, app_selection, ai_sampling_selection, thompson_selection, ots_selection, ucb_selection, bucb_selection
from bandits import learning_stationary as learning
from bandits import generative_process_swtch
from bandits import simulator

from multiprocessing import get_context

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

class Process:
    def __init__(self, rho=0.):
        self.log_pj_j = jnp.log(jnp.array([[1 - rho, rho], [1., 0.]]))
    def __call__(self, *args):
        return generative_process_swtch(*args, self.log_pj_j)
    
class Sim:
    def __init__(self, N, eps, steps):
        self.N = N
        self.eps = eps
        self.steps = steps
        self.lambdas =  jnp.arange(.0, 1.5, .015)


    def __call__(self, selection, K, seed):
        process = Process()
        if selection == 'TS':
            results = simulator(process, learning, thompson_selection, N=self.N, steps=self.steps, K=K, eps=self.eps, seed=seed)
        elif selection == 'OTS':
            results = simulator(process, learning, ots_selection, N=self.N, steps=self.steps, K=K, eps=self.eps, seed=seed)
        elif selection == 'UCB':
            results = simulator(process, learning, ucb_selection, N=self.N, steps=self.steps, K=K, eps=self.eps, seed=seed)
        elif selection == 'BUCB':
            results = simulator(process, learning, ucb_selection, N=self.N, steps=self.steps, K=K, eps=self.eps, seed=seed)
        elif selection == 'EFE':
            sim = lambda l: simulator(process, learning, lambda *args: efe_selection(*args, lam=l), N=self.N, steps=self.steps, K=K, eps=self.eps, seed=seed)
            results = vmap(sim)(self.lambdas)
        elif selection == 'APP':
            sim = lambda l: simulator(process, learning, lambda *args: app_selection(*args, lam=l), N=self.N, steps=self.steps, K=K, eps=self.eps, seed=seed)
            results = vmap(sim)(self.lambdas)
        elif selection == 'SMP':
            results = simulator(process, learning, ai_sampling_selection, N=self.N, steps=self.steps, K=K, eps=self.eps, seed=seed)
        return results, selection, K


def main(args):
    N = args.num_runs
    p = args.trial_power
    Ks = args.num_arms
    eps = args.difficulty/100.

    jax.config.update('jax_platform_name', args.device)

    steps = (p - 2) * 9
    seed = PRNGKey(10396145)

    regret_all = defaultdict(lambda: {})
    nargs = []
    for K in Ks:
        for name in ['TS', 'OTS', 'SMP', 'UCB', 'BUCB', 'EFE', 'APP']:
            seed, _seed = random.split(seed)
            nargs.append((name, K, _seed[0]))
    
    start = time()
    for _ in tqdm([1]):
        job = Sim(N, eps, steps)
        with get_context("spawn").Pool() as pool:
            for res, label, K in pool.starmap(job, nargs):
                print(label + '_' + str(K), time() - start)
                regret_all[label][K] = np.array(res).astype(np.float32)
        
    np.savez('data/stationary_Ks_e{}'.format(args.difficulty), **regret_all)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="All bandit algos in classical multi-armed bandits")
    parser.add_argument("-p", "--trial-power", nargs="?", default=3, type=int) # number of trials T = 10^p
    parser.add_argument("-n", "--num-runs", nargs='?', default=10, type=int)
    parser.add_argument("-k", "--num-arms", nargs='+', default=5, type=int)
    parser.add_argument("-d", "--difficulty", nargs='?', default=25, type=int)
    parser.add_argument('--device', nargs=1, default='gpu', type=str)

    args = parser.parse_args()

    main(args)
