# implementation of decision algorithms
import jax.numpy as jnp
from jax.scipy.special import digamma, betaln
from scipy.special import betaincinv
from jax import random

def thompson_selection(t, beliefs, rng_key):
    #Thompson sampling
    
    alpha_t = beliefs[..., 0]
    beta_t = beliefs[..., 1]
    
    thetas = random.beta(rng_key, alpha_t, beta_t)
    
    return random.categorical(rng_key, 1e3 * (2 * thetas - 1))

def ots_selection(t, beliefs, rng_key):
    #Optimistic thompson sampling
    alpha_t = beliefs[..., 0]
    beta_t = beliefs[..., 1]
    
    mu_t = alpha_t / (alpha_t + beta_t)
    
    thetas = random.beta(rng_key, alpha_t, beta_t)
    
    thetas = jnp.where(thetas > mu_t, thetas, mu_t) # keep values larger than mean
    
    return random.categorical(rng_key, 1e3 * (2 * thetas - 1))

def ucb_selection(t, beliefs, rng_key):
    # classical ucb algorithm
    alpha_t = beliefs[..., 0]
    beta_t = beliefs[..., 1]
    
    nu_t = alpha_t + beta_t
    mu_t = alpha_t/nu_t
    
    N, K = beliefs.shape[:-1]

    V = mu_t + jnp.sqrt(2 * jnp.log(1 + t)/(nu_t-2 + 1e-6))
        
    choices1 = random.categorical(rng_key, 1e3 * (V - V.mean(-1, keepdims=True)))
    choices2 = t * jnp.ones(N, dtype=jnp.int32)
    
    return jnp.where(t >= K, choices1, choices2)

def bucb_selection(t, beliefs, rng_key):
    # bayesian ucb algorithm 

    alpha_t = beliefs[..., 0]
    beta_t = beliefs[..., 1]
    
    perc = 1. - 1./(1. + t)
    Q = betaincinv(alpha_t, beta_t, perc)

    return random.categorical(rng_key, 1e3 * (2 * Q - 1))
    
def G(alpha_t, beta_t, alpha):
    nu_t = alpha_t + beta_t
    mu_t = alpha_t / nu_t
    
    KL_a = - betaln(alpha_t, beta_t) + (alpha_t - alpha) * digamma(alpha_t)\
             + (beta_t - 1) * digamma(beta_t) + (alpha + 1 - nu_t) * digamma(nu_t)
    
    H_a = - mu_t * digamma(alpha_t + 1) - (1-mu_t) * digamma(beta_t + 1) + digamma(nu_t + 1)
    
    return KL_a + H_a

def efe_selection(t, beliefs, rng_key, gamma=10, lam=1.):
    # expected free energy based action selection
    alpha_t = beliefs[..., 0]
    beta_t = beliefs[..., 1]
    
    alpha = jnp.exp(2 * lam)
    
    G_a = G(alpha_t, beta_t, alpha)
    
    choices = random.categorical(rng_key, - gamma * (G_a - G_a.mean(-1, keepdims=True))) # sample choices
    return choices

def S(alpha_t, beta_t, lam):
    
    nu_t = alpha_t + beta_t
    mu_t = alpha_t / nu_t
    
    KL_a = - lam * (2 * mu_t - 1) + mu_t * jnp.log(mu_t) + (1-mu_t) * jnp.log(1-mu_t) 
    
    H_a = - mu_t * digamma(alpha_t + 1) - (1-mu_t) * digamma(beta_t + 1) + digamma(nu_t + 1)
    
    return KL_a + H_a

def sup_selection(t, beliefs, rng_key, gamma=10., lam=1.):
    # expected surprisal based action selection
    alpha_t = beliefs[..., 0]
    beta_t = beliefs[..., 1]
    
    S_a = S(alpha_t, beta_t, lam) 
    
    choices = random.categorical(rng_key, - gamma * (S_a - S_a.mean(-1, keepdims=True))) # sample choices
    return choices

def app_selection(t, beliefs, rng_key, gamma=10., lam=1.):
    # expected surprisal based action selection
    alpha_t = beliefs[..., 0]
    beta_t = beliefs[..., 1]
    
    nu_t = alpha_t + beta_t
    mu_t = alpha_t/nu_t
    
    S_a = - lam * (2 * mu_t - 1) - 1/(2 * nu_t)
    
    choices = random.categorical(rng_key, - gamma * (S_a - S_a.mean(-1, keepdims=True))) # sample choices
    return choices
