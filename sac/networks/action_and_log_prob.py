import distrax as dx
import jax.nn as jnn
import jax.numpy as jnp

from rltools.types import Array, PRNGKey


def get_actions_log_probs(key: PRNGKey, dist: dx.Distribution) -> tuple[Array, Array]:
    """Returns SAC actions and log probs

    In the case of continuous action space, this returns actions and log probs
    after applying the Tanh bijection

    Args:
        key : A PRNGKey
        dist : A distrax.Distribution

    Returns:
        actions : A single or a batch of actions
        log_probs : A single or a batch of log_probs
    """
    actions, log_probs = dist.sample_and_log_prob(seed=key)

    if isinstance(dist, dx.Normal):
        actions = jnn.tanh(actions)
        log_probs -= jnp.log(1 - jnp.square(actions) + 1e-6)
        log_probs = jnp.sum(log_probs, axis=-1, keepdims=True)

    return actions, log_probs
