# methods for generative processes

import jax.numpy as jnp
from jax import random
from jax.scipy.special import logit, expit

def generative_process_swtch(t, choices, states, rng_key, log_pj_j):
    probs, changes = states

    N = len(choices)
    K = len(probs)

    rng_key, _rng_key = random.split(rng_key)
    new_change = random.categorical(_rng_key, log_pj_j[changes])

    rng_key, _rng_key = random.split(rng_key)
    random_probs = random.uniform(_rng_key, shape=(K,))

    new_probs = jnp.where(new_change, random_probs, probs)

    rng_key, _rng_key = random.split(rng_key)
    outcomes = random.bernoulli(_rng_key, probs, shape=(K,))

    return outcomes[choices], [new_probs, new_change]

def generative_process_drift(t, choices, states, rng_key, sigma=.01):
    probs, _ = states
    K = len(probs)
    
    rng_key, _rng_key = random.split(rng_key)
    x = logit(probs) + jnp.sqrt(sigma) * random.normal(_rng_key, shape=(K,))
    
    rng_key, _rng_key = random.split(rng_key)
    outcomes = random.bernoulli(_rng_key, probs)
    
    return outcomes[choices], [expit(x), jnp.zeros(K, dtype=jnp.int32)]
