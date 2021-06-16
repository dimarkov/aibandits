import argparse
import os
import timeit

import jax
import jax.numpy as jnp
from jax import random, jit
from multiprocessing import get_context

from numpy.core.fromnumeric import shape


# load functions
from bandits import learning_switching
from bandits import thompson_selection, ucb_selection, ots_selection, bucb_selection, efe_selection, app_selection

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

selection = {
    'TS': jit(thompson_selection),
    'UCB': jit(ucb_selection),
    'O-TS': jit(ots_selection),
    'B-UCB': jit(bucb_selection),
    'G-AI': jit(efe_selection),
    'A-AI': jit(app_selection),
}

def main(args):

    def run(func, *args):
        res = func(*args)
        res[0].block_until_ready()

    N = args.num_runs
    Ks = args.num_arms
    for K in Ks:
        beliefs = jnp.ones((N, K, 2))
        t = 100
        print("\nEstimate runtime of action selection algo ...")
        for name in args.algos:
            run(selection[name], t, beliefs, random.PRNGKey(0))
            print(name, "K={}".format(K), timeit.timeit(lambda: run(selection[name], t, beliefs, random.PRNGKey(0)), number=1000))
        
        print("\nEstimate runtime of the learning algo ...")
        outcomes = jnp.zeros((N, K))
        choices = jnp.zeros(N, dtype=jnp.int32)
        print('rho=0', "K={}".format(K), timeit.timeit(lambda: jit(learning_switching)(outcomes, choices, beliefs), number=100))
        print('rho>0', "K={}".format(K), timeit.timeit(lambda: jit(learning_switching)(outcomes, choices, beliefs, rho=.01), number=100))

        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Stationary multi-armed bandits with fixed difficulty")
    parser.add_argument("-p", "--trial-power", nargs='?', default=3, type=int) # number of trials T=10^p
    parser.add_argument("-n", "--num-runs", nargs='?', default=10, type=int)
    parser.add_argument("-k", "--num-arms", nargs='+', default=5, type=int)
    parser.add_argument("-d", "--difficulty", nargs='?', default='fixed', type=str)
    parser.add_argument("--algos", nargs='+', default=['TS', 'O-TS', 'UCB', 'B-UCB', 'G-AI', 'A-AI'], type=str)
    parser.add_argument("--device", nargs='?', default='gpu', type=str)

    args = parser.parse_args()
    main(args)
