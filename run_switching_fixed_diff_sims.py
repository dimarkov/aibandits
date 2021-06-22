import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from jax import random, vmap
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import get_context

# load functions
from bandits import learning_switching
from bandits import ots_selection, bucb_selection, efe_selection, app_selection
from bandits import sim_fixed_difficulty as simulator

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"


class Learning:
    def __init__(self, params):
        self.params = params
    
    def __call__(self, *args):
        return learning_switching(*args, **self.params)

class Selection:
    def __init__(self, selection, rho):
        self.selection = selection
        self.rho = rho

    def __call__(self, *args):
        return self.selection(*args, rho=self.rho)

class Sim:
    def __init__(self, N, T, save_every=100):
        steps = T//save_every
        self.params = {
            'N': N,
            'steps': steps,
            'save_every': save_every,
        }

        self.lambdas =  jnp.arange(0., 2.05, .05)

    def __call__(self, name, rho, eps, K, seed):
        self.params['rho'] = rho
        self.params['eps'] = eps
        self.params['seed'] = seed
        self.params['K'] = K
        self.params['linear'] = True
        learning = Learning(self.params)

        if name == 'O-TS':
            selection = Selection(ots_selection, rho)
            results = simulator(learning, selection, **self.params)
        elif name == 'B-UCB':
            jax.config.update('jax_platform_name', 'cpu')  # for some strange reason this speeds up mpi computations.
            selection = Selection(bucb_selection, rho)
            results = simulator(learning, selection, **self.params)
        elif name == 'G-AI':
            sim = lambda l: simulator(learning, lambda *args: efe_selection(*args, lam=l, rho=rho), **self.params)
            results = vmap(sim)(self.lambdas)
        elif name == 'A-AI':
            sim = lambda l: simulator(learning, lambda *args: app_selection(*args, lam=l), **self.params)
            results = vmap(sim)(self.lambdas)

        return results, name, self.params

def main(args):
    N = args.num_runs
    T = args.num_trials
    Ks = args.num_arms
    eps = args.difficulty/100
    
    jax.config.update('jax_platform_name', args.device)

    rng_key = random.PRNGKey(12345)
    regret_all = defaultdict(lambda: {})
    for name in tqdm(args.algos):
        nargs = []
        for rho in [0.005, .01, .02, .04]:
            R = int(1000 * rho)
            regret_all[name][R] = []
            for K in Ks:
                rng_key, _rng_key = random.split(rng_key)
                nargs.append((name, rho, eps, K, _rng_key[0]))

        job = Sim(N, T)
        with get_context("spawn").Pool() as pool:
            for res, name, params in pool.starmap(job, nargs):
                R = int(1000 * params['rho'])
                K = params['K']
                regret_all[name][R].append((K, np.array(res).astype(np.float32)))
            
        np.savez('data/tmp_switching_e{}_T{}_{}'.format(args.difficulty, T, name), regret_all[name])
    np.savez('data/switching_e{}_T{}'.format(args.difficulty, T), **regret_all)

    # delete tmp files
    for name in args.algos:
        os.remove('data/tmp_switching_e{}_T{}_{}.npz'.format(args.difficulty, T, name))
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="All bandit algos in switching multi-armed bandits")
    parser.add_argument("-t", "--num-trials", nargs='?', default=10000, type=int)
    parser.add_argument("-n", "--num-runs", nargs='?', default=1000, type=int)
    parser.add_argument("-k", "--num-arms", nargs='+', default=5, type=int)
    parser.add_argument("-d", "--difficulty", nargs='?', default=25, type=int)
    parser.add_argument("--algos", nargs='+', default=['O-TS', 'B-UCB', 'G-AI', 'A-AI'], type=str)
    parser.add_argument("--device", nargs='?', default='gpu', type=str)

    args = parser.parse_args()
    main(args)
