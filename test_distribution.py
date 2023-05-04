from sac.distributions import reparametrize_gaussian_and_tanh
import distrax as dx
import jax.numpy as jnp
import jax.random as jrd

key = jrd.PRNGKey(0)

loc = jnp.array([[0.0, 0.4, 7]])
log_scale = jnp.array([[-1.0, -7.0, 0.4]])

print(loc.shape, log_scale.shape)

sample, log_prob = reparametrize_gaussian_and_tanh(key, loc, log_scale, True)
print(sample)
