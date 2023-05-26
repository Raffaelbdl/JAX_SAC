import distrax as dx
import haiku as hk
import jax.numpy as jnp
import jax.random as jrd

import rltools.networks as nets
from rltools.types import Array


class AtariPolicyNetwork(hk.Module):
    """Discrete Policy Network for Atari environment"""

    def __init__(self, config: dict, name: str | None = None):
        super().__init__(name)
        self.n_actions = config["action_space"].n

    def __call__(self, observations: Array) -> tuple[Array, Array]:
        observations = observations.astype(jnp.float32)

        x = nets.AtariNatureCNN("b w h -> b w h 1", "nature_cnn")(observations)
        x = hk.Flatten()(x)
        x = nets.LinearEncoder([512], "encoder")(x)
        logits = hk.Linear(self.n_actions)(x)

        return self._get_action(logits)

    def _get_action(self, logits: Array) -> tuple[Array, Array]:
        return dx.Categorical(logits)


class AtariQValueNetwork(hk.Module):
    """Discrete QValue Network for Atari environment"""

    def __init__(self, config: dict, name: str | None = None):
        super().__init__(name)
        self.n_actions = config["action_space"].n

    def __call__(self, observations: Array) -> Array:
        observations = observations.astype(jnp.float32)

        x = nets.AtariNatureCNN("b w h -> b w h 1", "nature_cnn")(observations)
        x = hk.Flatten()(x)
        x = nets.LinearEncoder([512], "encoder")(x)

        return hk.Linear(self.n_actions)(x)


def get_atari_networks(config: dict):
    key1, key2, key3 = jrd.split(jrd.PRNGKey(config["seed"]), 3)
    observations = jnp.zeros((1,) + config["observation_space"].shape)

    @hk.transform
    def policy_transformed(observations: Array):
        return AtariPolicyNetwork(config, "actor")(observations)

    policy_fwd = policy_transformed.apply
    policy_params = policy_transformed.init(key1, observations)

    @hk.transform
    def qvalue_transformed(observations: Array):
        q1 = AtariQValueNetwork(config, "q1")(observations)
        q2 = AtariQValueNetwork(config, "q2")(observations)
        return q1, q2

    qvalue_fwd = qvalue_transformed.apply
    qvalue_params = qvalue_transformed.init(key2, observations)

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
