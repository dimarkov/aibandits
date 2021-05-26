# methods for generative processes

import jax.numpy as jnp
from jax import random, lax

def simulator(process, learning, action_selection, N=100, seed=0, steps=1000, K=10, eps=.25, save_every=100):
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
        res = lax.fori_loop(t * save_every, (t+1) * save_every, loop_fn, carry)
        _, _, _, cum_reg, cum_epst_reg = res
        return res, (cum_reg, cum_epst_reg)
    
    rng_key = random.PRNGKey(seed)
    probs = jnp.concatenate([jnp.array([eps + .5]), jnp.ones(K-1)/2.])
    states = [probs, jnp.zeros(1, dtype=jnp.int32)]
    prior = jnp.ones((N, K, 2))
    
    _, results = lax.scan(sim_fn, (rng_key, states, prior, jnp.zeros(N), jnp.zeros(N)), jnp.arange(steps))
    
    return results
