import argparse
import os
import timeit

import jax
import jax.numpy as jnp
from jax import random, jit, lax

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

learning = jit(learning_switching)

def main(args):

    def run(func, outcomes, beliefs):
        
        def scan_fun(carry, t):

            prior, rng_key = carry

            choices = func(t, prior, rng_key)

            posterior = learning(outcomes, choices, prior)

            return (posterior, rng_key), None

        rng_key = random.PRNGKey(0)

        last, _ = lax.scan(scan_fun, (beliefs, rng_key), jnp.arange(10000))

        last[0].block_until_ready()

    N = args.num_runs
    Ks = args.num_arms
    number = 10
    for K in Ks:
        outcomes = jnp.zeros((N, K))
        beliefs = jnp.ones((N, K, 2))

        print("\nEstimate runtime of bandit algorithms for K={}".format(K))
        for name in args.algos:
            run(selection[name], outcomes, beliefs)
            print(name, "K={}".format(K), timeit.timeit(lambda: run(selection[name], outcomes, beliefs), number=number)/(number*10))

        

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
