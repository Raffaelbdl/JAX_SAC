import distrax as dx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrd


def reparametrize_gaussian_and_tanh(key, loc, log_scale, return_log_prob):
    scale = jnp.exp(log_scale)
    sample, log_prob = dx.Normal(loc, scale).sample_and_log_prob(seed=key)

    tanh_sample = jnp.tanh(sample)
    if not return_log_prob:
        return tanh_sample

    tanh_log_prob = log_prob - jnp.log(1.0 - jnp.square(tanh_sample) + 1e-6)
    tanh_log_prob = jnp.sum(tanh_log_prob, axis=-1, keepdims=True)

    return tanh_sample, tanh_log_prob


def normal_and_tanh_sample_and_log_prob(key, normal_dist: dx.Normal):
    assert isinstance(
        normal_dist, dx.Normal
    ), "normal_dist is not a dx.Normal distribution"

    sample, log_prob = normal_dist.sample_and_log_prob(seed=key)

    tanh_sample = jnp.tanh(sample)
    tanh_log_prob = log_prob - jnp.log(1.0 - jnp.square(tanh_sample) + 1e-6)

    return tanh_sample, tanh_log_prob
