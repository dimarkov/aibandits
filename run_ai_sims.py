import jax.numpy as jnp
import numpy as np
import itertools
import argparse

from collections import defaultdict
from tqdm import trange
from jax import random, lax, vmap

from choice_algos import efe_selection, sup_selection, app_selection
from learning_algos import learning_stationary as learning
from environment import generative_process

rho = .0
log_pj_j = jnp.log(jnp.array([[1 - rho, rho],[1., 0.]]))
process = lambda *args: generative_process(*args, log_pj_j)

# simulator for POMDP
def simulator(process, learning, action_selection, N=100, T=1000, K=10, seed=0, eps=.25):
    def sim_fn(carry, t):
        rng_key, states, prior = carry

        rng_key, _rng_key = random.split(rng_key)
        choices = action_selection(t, prior, _rng_key)

        rng_key, _rng_key = random.split(rng_key)
        outcomes, states = process(t, choices, states, _rng_key)
        posterior = learning(outcomes, choices, prior)

        return (rng_key, states, posterior), choices

    rng_key = random.PRNGKey(seed)
    probs = jnp.concatenate([jnp.array([eps + .5]), jnp.ones(K-1)/2.])
    states = [probs, jnp.zeros(1, dtype=jnp.int32)]
    prior = jnp.ones((N, K, 2))/2

    _, choices = lax.scan(sim_fn, (rng_key, states, prior), jnp.arange(T))

    return choices

def main(args):
    N = args.num_runs
    T = args.num_trials
    times = np.arange(1, T + 1, 10)[:, None]
    Ks = args.num_arms
    eps = args.difficulty/100.

    gammas = jnp.arange(1., 21., 1.)
    lambdas = jnp.arange(.0, 4., .2)
    vals = jnp.array(list(itertools.product(gammas, lambdas)))
    for K in Ks:
        mean_reg = {}
        for func, label in zip([efe_selection, sup_selection, app_selection], ['EFE_K{}'.format(K), 'SUP_K{}'.format(K), 'APP_K{}'.format(K)]):
            sim = lambda g, l: simulator(process, learning, lambda *args: func(*args, gamma=g, lam=l), N=N, T=T, K=K, eps=eps)
            choices = vmap(sim)(vals[:, 0], vals[:, 1])
            regret = np.cumsum((1 - (choices == 0).astype(jnp.float32)) * eps, -2)[:, ::10]
            mean_reg[label] = regret/times

        np.savez('res_AI_K{}_e{}'.format(K, args.difficulty, **mean_reg))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Active inference algos in classical multi-armed bandits")
    parser.add_argument("-t", "--num-trials", nargs="?", default=100, type=int)
    parser.add_argument("-n", "--num-runs", nargs='?', default=10, type=int)
    parser.add_argument("-k", "--num-arms", nargs='+', default=5, type=int)
    parser.add_argument("-d", "--difficulty", nargs='?', default=25, type=int)

    args = parser.parse_args()

    main(args)
