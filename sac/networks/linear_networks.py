from typing import Callable

import distrax as dx
import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrd
import numpy as np

import rltools.networks as nets
from rltools.types import Array


class ContinuousPolicyNetwork(hk.Module):
    """Continuous Policy Network for environment with vector observations"""

    def __init__(self, config: dict, name: str | None = None):
        super().__init__(name)
        self.n_actions = np.prod(np.array(config["action_space"].shape))

        self.log_scale_min = config["log_scale_min"]
        self.log_scale_max = config["log_scale_max"]

    def __call__(self, observations: Array) -> tuple[Array, Array]:
        x = nets.LinearEncoder([128, 128], "encoder")(observations)

        locs = hk.Linear(self.n_actions)(x)
        log_scales = jnn.tanh(hk.Linear(self.n_actions)(x))
        log_scales = 0.5 * (self.log_scale_max - self.log_scale_min) * (log_scales + 1)
        log_scales += self.log_scale_min

        return dx.Normal(locs, jnp.exp(log_scales))


class DiscretePolicyNetwork(hk.Module):
    """Discrete Policy Network for environment with vector observations"""

    def __init__(self, config: dict, name: str | None = None):
        super().__init__(name)
        self.n_actions = config["action_space"].n

    def __call__(self, observations: Array) -> tuple[Array, Array]:
        x = nets.LinearEncoder([128, 128], "encoder")(observations)

        logits = hk.Linear(self.n_actions)(x)

        return dx.Categorical(logits)

    def _get_action(self, logits: Array) -> tuple[Array, Array]:
        return dx.Categorical(logits).sample_and_log_prob(seed=hk.next_rng_key())


class ContinuousQValueNetwork(hk.Module):
    """Continuous QValue Network for environment with vector observations"""

    def __init__(self, config: dict, name: str | None = None):
        super().__init__(name)

    def __call__(self, observations: Array, actions: Array) -> Array:
        obs_action_pairs = jnp.concatenate([observations, actions], axis=-1)

        x = nets.LinearEncoder([256, 256], "encoder")(obs_action_pairs)

        return hk.Linear(1)(x)


class DiscreteQValueNetwork(hk.Module):
    """Discrete QValue Network for environment with vector observations"""

    def __init__(self, config: dict, name: str | None = None):
        super().__init__(name)
        self.n_actions = config["action_space"].n

    def __call__(self, observations: Array) -> Array:
        x = nets.LinearEncoder([256, 256], "encoder")(observations)

        return hk.Linear(self.n_actions)(x)


def get_linear_networks(config: dict) -> tuple[list[Callable], list[hk.Params]]:
    """Returns linear networks for SAC

    Args:
        config : A dict that contains :
            - seed : int
            - observation_space : Box
            - action_space : Box | Discrete
            - discrete : bool

        if discrete == True, config needs the following items :
            - log_scale_min : float
            - log_scale_max : float

    Returns:
        fwd : A list of hk.Transformed apply
        params : A list of hk.Params
    """

    key1, key2, key3 = jrd.split(jrd.PRNGKey(config["seed"]), 3)

    observations = jnp.zeros((1,) + config["observation_space"].shape)

    @hk.transform
    def policy_transformed(observations: np.ndarray):
        if config["discrete"]:
            return DiscretePolicyNetwork(config, "policy")(observations)
        return ContinuousPolicyNetwork(config, "policy")(observations)

    policy_fwd = policy_transformed.apply
    policy_params = policy_transformed.init(key1, observations)

    qvalue_fwd, qvalue_params = get_qvalue_networks(key2, config)

    @hk.transform
    def alpha_transformed():
        return hk.get_parameter(
            "alpha", (), jnp.float32, hk.initializers.Constant(-3.5)
        )

    alpha_fwd = alpha_transformed.apply
    alpha_params = alpha_transformed.init(key3)

    fwd = [policy_fwd, qvalue_fwd, alpha_fwd]
    params = [policy_params, qvalue_params, alpha_params]
    return fwd, params


def get_qvalue_networks(key, config: dict) -> tuple[Callable, hk.Params]:
    """Returns linear qvalue networks for SAC"""

    observations = jnp.zeros((1,) + config["observation_space"].shape)
    actions = jnp.zeros((1,) + config["action_space"].shape)

    inputs = [observations] if config["discrete"] else [observations, actions]
    qvalue_network = (
        DiscreteQValueNetwork if config["discrete"] else ContinuousQValueNetwork
    )

    @hk.transform
    def qvalue_transformed(*inputs):
        q1 = qvalue_network(config, "q1")(*inputs)
        q2 = qvalue_network(config, "q2")(*inputs)
        return q1, q2

    qvalue_fwd = qvalue_transformed.apply
    qvalue_params = qvalue_transformed.init(key, *inputs)

    return qvalue_fwd, qvalue_params
