# methods for generative processes

import jax.numpy as jnp
from jax import random
from jax.scipy.special import logit, expit

__all__ = [
    'generative_process_swtch1',
    'generative_process_swtch2',
    'generative_process_drift'
]

def generative_process_swtch1(t, choices, states, rng_key, logits, **kwargs):
    """Switching generative process where the highest reward probability p=\epsilon + 1/2
    changes to another arm with probability \rho or remains on the same arm with probability 1 - \rho.
    """
    N = kwargs['N']
    K = kwargs['K']
    eps = kwargs['eps']
    
    rng_key, _rng_key = random.split(rng_key)
    new_states = random.categorical(_rng_key, logits[states])
    
    rng_key, _rng_key = random.split(rng_key)
    reward_probs = .5 + eps * jnp.eye(K)[states]
    outcomes = random.bernoulli(_rng_key, reward_probs, shape=(N, K))
    
    return outcomes, new_states


def generative_process_swtch2(t, choices, states, rng_key, logits, **kwargs):
    """Switching generative process where the reward probability associated with each 
    arm either stay the same with probaiblity 1 - \rho, or change with probability \rho.
    In the case of a change the probabilities on all arms are sampled from a uniform 
    distribution.
    """
    probs, changes = states

    N, K = probs.shape

    assert len(changes) == N

    rng_key, _rng_key = random.split(rng_key)
    new_change = random.categorical(_rng_key, logits[changes])

    rng_key, _rng_key = random.split(rng_key)
    random_probs = random.uniform(_rng_key, shape=(N, K))

    new_probs = jnp.where(jnp.expand_dims(new_change, -1), random_probs, probs)

    rng_key, _rng_key = random.split(rng_key)
    outcomes = random.bernoulli(_rng_key, probs)

    assert outcomes.shape == (N, K)

    return outcomes, [new_probs, new_change]


def generative_process_drift(t, choices, states, rng_key, sigma=.01, **kwargs):
    """Drifting generative process where probabilities associated with each 
    arm follow independently a random walk in the logit space.
    """
    probs, _ = states
    N = len(choices)
    K = len(probs)
    
    rng_key, _rng_key = random.split(rng_key)
    x = logit(probs) + jnp.sqrt(sigma) * random.normal(_rng_key, shape=(K,))
    
    rng_key, _rng_key = random.split(rng_key)
    outcomes = random.bernoulli(_rng_key, probs, shape=(N, K))
    
    return outcomes, [expit(x), jnp.zeros(K, dtype=jnp.int32)]
