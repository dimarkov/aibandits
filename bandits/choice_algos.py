# implementation of decision algorithms
import jax.numpy as jnp
from jax.scipy.special import digamma
from jax import random
from .utils import betaincinv

__all__ = [
    'thompson_selection',
    'ots_selection',
    'ucb_selection',
    'bucb_selection',
    'efe_selection',
    'app_selection',
    'ai_sampling_selection'
]

# choice algos for stationary bandits
def thompson_selection(t, beliefs, rng_key):
    #Thompson sampling
    
    alpha_t = beliefs[..., 0]
    beta_t = beliefs[..., 1]
    
    thetas = random.beta(rng_key, alpha_t, beta_t)
    
    return jnp.argmax(thetas,  -1)

def ots_selection(t, beliefs, rng_key):
    #Optimistic thompson sampling
    alpha_t = beliefs[..., 0]
    beta_t = beliefs[..., 1]
    
    mu_t = alpha_t / (alpha_t + beta_t)
    
    thetas = random.beta(rng_key, alpha_t, beta_t)
    
    thetas = jnp.where(thetas > mu_t, thetas, mu_t) # keep values larger than mean
    
    return jnp.argmax(thetas,  -1)

def ucb_selection(t, beliefs, rng_key):
    # classical ucb algorithm
    alpha_t = beliefs[..., 0]
    beta_t = beliefs[..., 1]
    
    n_t = alpha_t + beta_t - 2 + 1e-6
    mu_t = (alpha_t - 1)/n_t
    
    N, K = beliefs.shape[:-1]

    lnt = jnp.log(t + 1)
    V = mu_t + lnt/n_t + jnp.sqrt(mu_t * lnt/n_t)
        
    choices1 = random.categorical(rng_key, 1e5 * V)
    choices2 = t * jnp.ones(N, dtype=jnp.int32)
    
    return jnp.where(t >= K, choices1, choices2)

def bucb_selection(t, beliefs, rng_key):
    # bayesian ucb algorithm 

    alpha_t = beliefs[..., 0]
    beta_t = beliefs[..., 1]
    
    perc = 1. - 1./(1 + t)
    Q = betaincinv(alpha_t, beta_t, perc)

    return random.categorical(rng_key, 1e5 * (2 * Q - 1))
    
def G(alpha_t, beta_t, lam):
    
    nu_t = alpha_t + beta_t
    mu_t = alpha_t / nu_t
    
    KL_a = - lam * (2 * mu_t - 1) + mu_t * jnp.log(mu_t) + (1-mu_t) * jnp.log(1-mu_t) 
    
    H_a = - mu_t * digamma(alpha_t + 1) - (1-mu_t) * digamma(beta_t + 1) + digamma(nu_t + 1)
    
    return KL_a + H_a

def efe_selection(t, beliefs, rng_key, gamma=1e5, lam=1.):
    # expected surprisal based action selection
    alpha_t = beliefs[..., 0]
    beta_t = beliefs[..., 1]
    
    S_a = G(alpha_t, beta_t, lam) 
    
    choices = random.categorical(rng_key, - gamma * S_a) # sample choices
    return choices

def app_selection(t, beliefs, rng_key, gamma=1e5, lam=1.):
    # expected surprisal based action selection
    alpha_t = beliefs[..., 0]
    beta_t = beliefs[..., 1]
    
    nu_t = alpha_t + beta_t
    mu_t = alpha_t/nu_t
    
    S_a = - lam * (2 * mu_t - 1) - 1/(2 * nu_t)
    
    choices = random.categorical(rng_key, - gamma * S_a) # sample choices
    return choices

def ai_sampling_selection(t, beliefs, rng_key, lam=1.):
    alpha_t = beliefs[..., 0]
    beta_t = beliefs[..., 1]

    nu_t = alpha_t + beta_t
    mu_t = alpha_t/nu_t

    thetas = random.beta(rng_key, alpha_t, beta_t)

    KL_a = - lam * (2 * thetas - 1) + thetas * jnp.log(mu_t) + (1 - thetas) * jnp.log(1 - mu_t)
    H_a = - thetas * jnp.log(thetas) - (1 - thetas) * jnp.log(1 - thetas)

    G_a = KL_a + H_a

    return jnp.argmin(G_a, -1)