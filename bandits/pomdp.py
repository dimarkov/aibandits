# methods for generative processes

import jax.numpy as jnp
from jax import lax, random

__all__ = [
    'simulator',
    'generate_checkpoints'
]

def generate_checkpoints(save_every, steps):
    checks = [0, save_every]
    for t in range(steps):
        checks.append(save_every)
        save_every = jnp.where((t + 1) % 9 == 0, save_every * 10, save_every)

    return jnp.cumsum(jnp.array(checks))

def simulator(process, learning, action_selection, N=100, seed=0, steps=45, K=10, eps=.25):
    # save checkpoints in logarithmic scale
    checks = generate_checkpoints(100, steps)

    def loop_fn(t, carry):
        rng_key, states, prior, cum_reg = carry

        rng_key, _rng_key = random.split(rng_key)
        choices = action_selection(t, prior, _rng_key)

        rng_key, _rng_key = random.split(rng_key)
        outcomes, states = process(t, choices, states, _rng_key)
        posterior = learning(outcomes, choices, prior)

        sel = jnp.arange(N)
        cum_reg += eps * ~(choices == 0)

        # compute epistemic regret
        # alphas = prior[sel, choices, 0]
        # betas = prior[sel, choices, 1]
        # nu = alphas + betas
        # mu = alphas/nu

        # nu_min = jnp.min(prior[..., 0] + prior[..., 1], -1)
        # cum_epst_reg += (1/nu_min - 1/nu)/2

        return (rng_key, states, posterior, cum_reg)
    
    def sim_fn(carry, t):
        res = lax.fori_loop(checks[t], checks[t+1], loop_fn, carry)
        
        return res, res[-1]
    
    probs = jnp.concatenate([jnp.array([eps + .5]), jnp.ones(K-1)/2.])
    states = [probs, jnp.zeros(1, dtype=jnp.int32)]
    prior = jnp.ones((N, K, 2))
    
    _, results = lax.scan(sim_fn, (random.PRNGKey(seed), states, prior, jnp.zeros(N)), jnp.arange(steps))
    
    return results
