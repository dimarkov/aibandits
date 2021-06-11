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
from bandits import thompson_selection, ucb_selection, ots_selection, bucb_selection, efe_selection, app_selection, sai_selection
from bandits import sim_fixed_difficulty, sim_varying_difficulty

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

selection = {
    'TS': thompson_selection,
    'UCB': ucb_selection,
    'O-TS': ots_selection,
    'B-UCB': bucb_selection,
    'G-AI': efe_selection,
    'A-AI': app_selection,
    'S-AI': sai_selection
}


class Learning:
    def __init__(self, params):
        self.params = params
    
    def __call__(self, *args):
        return learning_switching(*args, **self.params)


class Sim:
    def __init__(self, N, steps, save_every=100, fixed=True):
        self.params = {
            'N': N,
            'steps': steps,
            'save_every': save_every,
        }

        if fixed:
            self.simulator = sim_fixed_difficulty
        else:
            self.simulator = sim_varying_difficulty

        self.lambdas =  jnp.arange(0., 2.05, .05)
        self.params['rho'] = 0.

    def __call__(self, name, eps, K, seed):
        self.params['eps'] = eps
        self.params['seed'] = seed
        self.params['K'] = K
        learning = Learning(self.params)

        if name in ['TS', 'O-TS', 'UCB']:
            results = self.simulator(learning, selection[name], **self.params)
        elif name == 'B-UCB':
            jax.config.update('jax_platform_name', 'cpu')  # for some strange reason this speeds up mpi computations.
            results = self.simulator(learning, bucb_selection, **self.params)
        elif name in ['G-AI', 'A-AI', 'S-AI']:
            sim = lambda l: self.simulator(learning, lambda *args: selection[name](*args, lam=l), **self.params)
            results = vmap(sim)(self.lambdas)
 
        return results, name, self.params


def main(args):

    N = args.num_runs
    p = args.trial_power
    Ks = args.num_arms
    if args.difficulty == 'varying':
        fixed = False
    else:
        fixed = True
    
    steps = (p - 2) * 9

    rng_key = random.PRNGKey(12345)
    regret_all = defaultdict(lambda: {})
    for name in tqdm(args.algos):
        nargs = []
        for eps in [.05, .1, .2]:
            E = int(100 * eps)
            regret_all[name][E] = []
            for K in Ks:
                rng_key, _rng_key = random.split(rng_key)
                nargs.append((name, eps, K, _rng_key[0]))

        job = Sim(N, steps, fixed=fixed)
        with get_context("spawn").Pool() as pool:
            for res, name, params in pool.starmap(job, nargs):
                E = int(100 * params['eps'])
                K = params['K']
                regret_all[name][E].append((K, np.array(res).astype(np.float32)))
            
        np.savez('data/tmp_stationary_{}_diff_T{}_{}'.format(args.difficulty, p, name), regret_all[name])
    
    if len(args.algos) > 1:
        np.savez('data/stationary_{}_diff_T{}'.format(args.difficulty, p), **regret_all)
    else:
        np.savez('data/stationary_{}_diff_T{}_{}'.format(args.difficulty, p, name), **regret_all)

    # delete tmp files
    for name in args.algos:
        os.remove('data/tmp_stationary_{}_diff_T{}_{}.npz'.format(args.difficulty, p, name))
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Stationary multi-armed bandits with fixed difficulty")
    parser.add_argument("-p", "--trial-power", nargs='?', default=3, type=int) # number of trials T=10^p
    parser.add_argument("-n", "--num-runs", nargs='?', default=10, type=int)
    parser.add_argument("-k", "--num-arms", nargs='+', default=5, type=int)
    parser.add_argument("-d", "--difficulty", nargs='?', default='fixed', type=str)
    parser.add_argument("--algos", nargs='+', default=['TS', 'O-TS', 'UCB', 'B-UCB', 'G-AI', 'A-AI'], type=str)

    args = parser.parse_args()
    main(args)
