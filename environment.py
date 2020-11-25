# methods for generative processes

import jax.numpy as jnp
from jax import random

def generative_process(t, choices, states, rng_key, log_pj_j):
    probs, changes = states

    N = len(choices)
    K = len(probs)

    rng_key, _rng_key = random.split(rng_key)
    new_change = random.categorical(_rng_key, log_pj_j[changes])

    rng_key, _rng_key = random.split(rng_key)
    random_probs = random.uniform(_rng_key, shape=(K,))

    new_probs = jnp.where(new_change, random_probs, probs)

    rng_key, _rng_key = random.split(rng_key)
    outcomes = random.bernoulli(_rng_key, probs, shape=(N, K))

    return outcomes, [new_probs, new_change]
