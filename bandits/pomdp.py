# methods for generative processes

import jax.numpy as jnp
from jax import lax, random
from .environment import generative_process_swtch1, generative_process_swtch2

__all__ = [
    'sim_fixed_difficulty',
    'sim_varying_difficulty',
    'generate_log_checkpoints',
    'generate_lin_checkpoints'
]

class ProcessFixed:
    def __init__(self, params):
        self.params = params
        K = params['K']  # number of arms
        rho = params['rho']  # change probability
        self.logits = jnp.log((1 - rho) * jnp.eye(K) + rho * (jnp.ones((K, K)) - jnp.eye(K))/(K-1))

    def __call__(self, *args):
        return generative_process_swtch1(*args, self.logits, **self.params)

class ProcessVarying:
    def __init__(self, params):
        rho = params['rho']  # change probability
        self.logits = jnp.log(jnp.array([[1 - rho, rho], [1, 0]]))

    def __call__(self, *args):
        return generative_process_swtch2(*args, self.logits)

def generate_log_checkpoints(steps, save_every=100):
    checks = [0, save_every]
    for t in range(steps):
        checks.append(save_every)
        save_every = jnp.where((t + 1) % 9 == 0, save_every * 10, save_every)

    return jnp.cumsum(jnp.array(checks))

def generate_lin_checkpoints(steps, save_every=100):
    return jnp.arange(0, (steps + 1) * save_every, save_every, dtype=jnp.int32)

def sim_fixed_difficulty(learning, action_selection, N=100, seed=0, steps=45, save_every=100, K=10, eps=.25, rho=0., linear=False, **kwargs):

    # save checkpoints in logarithmic or linear scale
    # we use linear scale for dynamic bandits and logarithmic for stationary bandits
    if linear:
        checks = generate_lin_checkpoints(steps, save_every)
    else:
        checks = generate_log_checkpoints(steps, save_every)
        steps += 1

    process = ProcessFixed({'N': N, 'K': K, 'eps': eps, 'rho': rho})

    def loop_fn(t, carry):
        rng_key, states, prior, cum_reg = carry

        rng_key, _rng_key = random.split(rng_key)
        choices = action_selection(t, prior, _rng_key)

        rng_key, _rng_key = random.split(rng_key)
        outcomes, states = process(t, choices, states, _rng_key)
        posterior = learning(outcomes, choices, prior)

        cum_reg += eps * ~(choices == states)

        return (rng_key, states, posterior, cum_reg)
    
    def sim_fn(carry, t):
        res = lax.fori_loop(checks[t], checks[t+1], loop_fn, carry)
        
        return res, res[-1]
    
    rng_key, _rng_key = random.split(random.PRNGKey(seed))

    states = random.categorical(_rng_key, jnp.zeros(K), shape=(N,))
    prior = jnp.ones((N, K, 2))
    
    _, results = lax.scan(sim_fn, (rng_key, states, prior, jnp.zeros(N)), jnp.arange(steps))
    
    return results

def sim_varying_difficulty(learning, action_selection, N=100,  K=10, rho=0, seed=0, steps=45, save_every=100, linear=False, **kwargs):

    # save checkpoints in logarithmic scale
    if linear:
        checks = generate_lin_checkpoints(steps, save_every)
    else:
        checks = generate_log_checkpoints(steps, save_every)
        steps += 1

    process = ProcessVarying({'rho': rho})

    def loop_fn(t, carry):
        rng_key, states, prior, cum_reg = carry

        rng_key, _rng_key = random.split(rng_key)
        choices = action_selection(t, prior, _rng_key)

        rng_key, _rng_key = random.split(rng_key)
        outcomes, states = process(t, choices, states, _rng_key)
        posterior = learning(outcomes, choices, prior)

        cum_reg += states[0].max(-1) - states[0][jnp.arange(len(choices)), choices]

        return (rng_key, states, posterior, cum_reg)
    
    def sim_fn(carry, t):
        res = lax.fori_loop(checks[t], checks[t+1], loop_fn, carry)
        
        return res, res[-1]
    
    rng_key, _rng_key = random.split(random.PRNGKey(seed))

    states = [random.uniform(_rng_key, shape=(N, K)), jnp.zeros(N, dtype=jnp.int32)]
    prior = jnp.ones((N, K, 2))
    
    _, results = lax.scan(sim_fn, (rng_key, states, prior, jnp.zeros(N)), jnp.arange(steps))
    
    return results