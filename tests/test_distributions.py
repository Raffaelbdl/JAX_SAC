import distrax as dx
import jax.numpy as jnp
import jax.random as jrd
import numpy as np

from sac.distributions import normal_and_tanh_sample_and_log_prob


def test_normal_and_tanh_sample_and_log_prob():
    key = jrd.PRNGKey(0)

    def test_output_shape():
        shape = (5, 3, 6)
        loc = np.ones((shape))
        scale = np.ones((shape))

        dist = dx.Normal(loc, scale)
        sample, log_prob = normal_and_tanh_sample_and_log_prob(key, dist)

        assert sample.shape == shape
        assert log_prob.shape == shape

        summed_log_prob = np.sum(log_prob, axis=-1, keepdims=True)
        assert summed_log_prob.shape == shape[:-1] + (1,)

    def test_values_in_range():
        shape = (5, 3, 6)
        loc = 10 * np.ones((shape))
        scale = np.ones((shape))

        dist = dx.Normal(loc, scale)
        sample, log_prob = normal_and_tanh_sample_and_log_prob(key, dist)

        assert np.where(np.abs(sample) <= 1, True, False).all()

    test_output_shape()
    test_values_in_range()
