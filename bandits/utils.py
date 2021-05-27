# methods for different learning rules
import jax.numpy as jnp

from jax import jit
from jax.scipy.special import betaln, betainc
from opt_einsum import contract

def einsum(*args):
    return contract(*args, backend="jax")

@jit
def update_x(x, a, b, p, a1, b1, afac):
    err = betainc(a, b, x) - p
    t = jnp.exp(a1 * jnp.log(x) + b1 * jnp.log(1.0 - x) + afac)
    u = err/t
    tmp = u * (a1 / x - b1 / (1.0 - x))
    t = u/(1.0 - 0.5 * jnp.clip(tmp, a_max=1.0))
    x -= t
    x = jnp.where(x <= 0., 0.5 * (x + t), x)
    x = jnp.where(x >= 1., 0.5 * (x + t + 1.), x)
    
    return x, t

@jit
def func_1(a, b, p):
    pp = jnp.where(p < .5, p, 1. - p)
    t = jnp.sqrt(-2. * jnp.log(pp))
    x = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t
    x = jnp.where(p < .5, -x, x)
    al = (jnp.power(x, 2) - 3.0) / 6.0
    h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0))
    w = (x * jnp.sqrt(al + h) / h)-(1.0 / (2.0 * b - 1) - 1.0/(2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h))
    return a / (a + b * jnp.exp(2.0 * w))

@jit
def func_2(a, b, p):
    lna = jnp.log(a / (a + b))
    lnb = jnp.log(b / (a + b))
    t = jnp.exp(a * lna) / a
    u = jnp.exp(b * lnb) / b
    w = t + u

    return jnp.where(p < t/w, jnp.power(a * w * p, 1.0 / a), 1. - jnp.power(b *w * (1.0 - p), 1.0/b))

@jit
def compute_x(p, a, b):
    return jnp.where(jnp.logical_and(a >= 1.0, b >= 1.0), func_1(a, b, p), func_2(a, b, p))

@jit
def betaincinv(a, b, p):
    '''Example found at https://malishoaib.wordpress.com/2014/05/30/inverse-of-incomplete-beta-function-computational-statisticians-wet-dream/
    and adapted for jax jit compilation'''
     
    a1 = a - 1.0
    b1 = b - 1.0

    ERROR = 1e-8

    p = jnp.clip(p, a_min=0., a_max=1.)

    x = jnp.where(jnp.logical_or(p <= 0.0, p >= 1.), p, compute_x(p, a, b))
 
    afac = - betaln(a, b)
    stop  = jnp.logical_or(x == 0.0, x == 1.0)
    for i in range(10):
        x_new, t = update_x(x, a, b, p, a1, b1, afac)
        x = jnp.where(stop, x, x_new)
        stop = jnp.where(jnp.logical_or(jnp.abs(t) < ERROR * x, stop), True, False)

    return x

