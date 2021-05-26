# methods for different learning rules
import jax.numpy as jnp

def learning_stationary(outcomes, choices, priors):

    N, K = outcomes.shape

    select_observed = jnp.eye(K)[choices]

    alpha_t = priors[..., 0] + select_observed * outcomes
    beta_t = priors[..., 1] + select_observed * (1 - outcomes)

    return jnp.stack([alpha_t, beta_t], -1)
