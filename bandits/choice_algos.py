# implementation of decision algorithms
import jax.numpy as jnp
from jax.scipy.special import digamma
from jax import random, lax, jit
from .utils import betaincinv

__all__ = [
    'random_choices',
    'thompson_selection',
    'ots_selection',
    'ucb_selection',
    'bucb_selection',
    'efe_selection',
    'app_selection',
    'sai_selection'
]

def random_choices(t, beliefs, rng_key, **kwargs):
    N, K = beliefs.shape[:-1]
    return random.categorical(rng_key, jnp.zeros((N, K)))

def sample_switching(pars):
    _rng_key, rng_key = random.split(pars[0])
    binary = random.bernoulli(_rng_key, p=1.-pars[-1])
        
    _rng_key, rng_key = random.split(rng_key)
    theta_sample1 = random.beta(_rng_key, pars[1], pars[2])
    theta_sample2 = random.beta(_rng_key, 1., 1., shape=theta_sample1.shape)
    
    return jnp.where(binary, theta_sample1, theta_sample2)

def sample_stationary(pars):
    return random.beta(pars[0], pars[1], pars[2])

def thompson_selection(t, beliefs, rng_key, rho=0):
    #Thompson sampling
    
    alpha_t = beliefs[..., 0]
    beta_t = beliefs[..., 1]

    thetas = lax.cond(rho > 0., sample_switching, sample_stationary, (rng_key, alpha_t, beta_t, rho)) 

    return jnp.argmax(thetas,  -1)

def ots_selection(t, beliefs, rng_key, rho=0.):
    #Optimistic thompson sampling
    alpha_t = beliefs[..., 0]
    beta_t = beliefs[..., 1]
    
    mu_t = (1 - rho) * alpha_t/(alpha_t + beta_t) + rho/2

    thetas = lax.cond(rho > 0., sample_switching, sample_stationary, (rng_key, alpha_t, beta_t, rho)) 
    
    thetas = jnp.clip(thetas, a_min=mu_t) # keep values larger than mean
    
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

def bucb_selection(t, beliefs, rng_key, rho=0.):
    # bayesian ucb algorithm 

    bar_alpha = (1 - rho) * beliefs[..., 0] + rho
    bar_beta = (1 - rho) * beliefs[..., 1] + rho

    perc = 1. - 1./(1 + t)
    Q = betaincinv(bar_alpha, bar_beta, perc)

    return random.categorical(rng_key, 1e5 * (2 * Q - 1))  # sample_choices

def G(alpha_t, beta_t, lam, rho):
    
    nu_t = alpha_t + beta_t
    mu_t = alpha_t / nu_t
    tilde_mu = mu_t + rho * (.5 - mu_t)
    
    KL_a = - 2 * lam * (1 - rho) * mu_t + tilde_mu * jnp.log(tilde_mu) + (1 - tilde_mu) * jnp.log(1 - tilde_mu) 
    
    H_a = - mu_t * digamma(alpha_t + 1) - (1-mu_t) * digamma(beta_t + 1) + digamma(nu_t + 1)
    
    return KL_a + (1 - rho) * H_a

def efe_selection(t, beliefs, rng_key, gamma=1e5, lam=1., rho=0.):
    # expected surprisal based action selection
    alpha_t = beliefs[..., 0]
    beta_t = beliefs[..., 1]
    
    S_a = G(alpha_t, beta_t, lam, rho) 
    
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