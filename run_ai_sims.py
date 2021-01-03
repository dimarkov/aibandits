import jax.numpy as jnp
import numpy as np
import itertools
import argparse
import os 
import glob

from collections import defaultdict
from tqdm import trange
from jax import random, lax, vmap

from choice_algos import efe_selection, sup_selection, app_selection
from learning_algos import learning_stationary as learning
from environment import generative_process_swtch

rho = .0
log_pj_j = jnp.log(jnp.array([[1 - rho, rho], [1., 0.]]))
process = lambda *args: generative_process_swtch(*args, log_pj_j)

# simulator for POMDP
def simulator(process, learning, action_selection, N=100, T=1000, K=10, eps=.25, save_every=100):
    def loop_fn(t, carry):
        rng_key, states, prior, cum_reg, cum_epst_reg = carry

        rng_key, _rng_key = random.split(rng_key)
        choices = action_selection(t, prior, _rng_key)

        rng_key, _rng_key = random.split(rng_key)
        outcomes, states = process(t, choices, states, _rng_key)
        posterior = learning(outcomes, choices, prior)

        sel = jnp.arange(N)

        alphas = prior[sel, choices, 0]
        betas = prior[sel, choices, 1]
        nu = alphas + betas
        mu = alphas/nu

        nu_min = jnp.min(prior[..., 0] + prior[..., 1], -1)
        cum_reg += eps * ~(choices == 0)

        cum_epst_reg += (1/nu_min - 1/nu)/2

        return (rng_key, states, posterior, cum_reg, cum_epst_reg)

    def sim_fn(carry, t):
        res = lax.fori_loop(t*save_every, (t+1)*save_every, loop_fn, carry)
        _, _, _, cum_reg, cum_epst_reg = res
        return res, (cum_reg, cum_epst_reg)

    seed = 0

    rng_key = random.PRNGKey(seed)
    probs = jnp.concatenate([jnp.array([eps + .5]), jnp.ones(K-1)/2.])
    states = [probs, jnp.zeros(1, dtype=jnp.int32)]
    prior = jnp.ones((N, K, 2))

    _, results = lax.scan(sim_fn, (rng_key, states, prior, jnp.zeros(N), jnp.zeros(N)), jnp.arange(T))

    return results

def merge_files(args):
    merged_res = {}
    for K in args.num_arms:
        tmp = np.load('tmp_res_AI_K{}_e{}.npz'.format(K, args.difficulty), allow_pickle=True)
        for key in tmp.keys():
            merged_res[key] = tmp[key]
    np.savez('res_AI_Ks_e{}'.format(args.difficulty), **merged_res)
    # delete tmp files
    files = glob.glob(os.path.join('tmp_*_e{}.npz'.format(args.difficulty)))
    for file in files:
        os.remove(file)

def main(args):
    N = args.num_runs
    T = args.num_trials
    Ks = args.num_arms
    eps = args.difficulty/100.

    gammas = jnp.ones(1) * 1000.
    lambdas = jnp.arange(.0, 1.5, .015)
    vals = jnp.array(list(itertools.product(gammas, lambdas)))

    for K in Ks:
        regret_all = defaultdict()
        for func, label in zip([efe_selection, sup_selection, app_selection], ['EFE_K{}'.format(K), 'SUP_K{}'.format(K), 'APP_K{}'.format(K)]):
            sim = lambda g, l: simulator(process, learning, lambda *args: func(*args, gamma=g, lam=l), N=N, T=T, K=K, eps=eps)
            results = vmap(sim)(vals[:, 0], vals[:, 1])
            cum_regret = np.array(results[0]).astype(np.float32)
            cum_epst_regret = np.array(results[1]).astype(np.float32)
            regret_all[label] = {'regret': cum_regret, 'epistemic': cum_epst_regret}
        np.savez('tmp_res_AI_K{}_e{}'.format(K, args.difficulty), **regret_all)
    merge_files(args)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Active inference algos in classical multi-armed bandits")
    parser.add_argument("-t", "--num-trials", nargs="?", default=100, type=int)
    parser.add_argument("-n", "--num-runs", nargs='?', default=10, type=int)
    parser.add_argument("-k", "--num-arms", nargs='+', default=5, type=int)
    parser.add_argument("-d", "--difficulty", nargs='?', default=25, type=int)

    args = parser.parse_args()

    main(args)
