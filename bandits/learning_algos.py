# methods for different learning rules
import jax.numpy as jnp
from jax import jit, lax

__all__ = [
    'learning_stationary',
    'learning_switching'
]

@jit
def learning_stationary(outcomes, choices, priors):
    return learning_switching(outcomes, choices, priors)

@jit
def true_fun(outcomes, select_obs, alpha, beta, rho):
    mu = alpha/(alpha + beta)
    _lkl = outcomes * mu + (1 - outcomes) * (1 - mu)
    lkl = (_lkl * select_obs).sum(-1, keepdims=True)
    return .5 * rho /(.5 * rho + lkl * (1 - rho))

@jit
def false_fun(rho):
    return rho

@jit
def learning_switching(outcomes, choices, priors, rho=.0, **kwargs):
    N, K = outcomes.shape
    
    select_obs = jnp.eye(K)[choices]
    
    omega_new = lax.cond(rho > 0, 
                         lambda r: true_fun(outcomes, select_obs, priors[..., 0], priors[..., 1], r), 
                         false_fun, 
                         rho * jnp.ones((N, 1)))
                        
    omega_new = jnp.expand_dims(omega_new, -1)
    select_obs = jnp.expand_dims(select_obs, -1)

    posterior = (1 - omega_new) * priors + omega_new + select_obs * jnp.stack([outcomes, 1 - outcomes], -1)
    
    return posterior