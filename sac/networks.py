from typing import Optional, Type

import distrax
import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrd
import numpy as np


class ContinuousActor(hk.Module):
    def __init__(self, config: dict, name: Optional[str] = None):
        super().__init__(name)

        self.n = config["n_actions"]

        self.encoder_w_init = hk.initializers.Orthogonal(np.sqrt(2))
        self.output_w_init = hk.initializers.Orthogonal(0.01)
        self.b_init = hk.initializers.Constant(0.0)

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = observations

        x = jnn.relu(hk.Linear(256, w_init=self.encoder_w_init, b_init=self.b_init)(x))
        x = jnn.relu(hk.Linear(256, w_init=self.encoder_w_init, b_init=self.b_init)(x))

        m_logits = hk.Linear(self.n, w_init=self.output_w_init, b_init=self.b_init)(x)
        m_std_logs = jnn.tanh(
            hk.Linear(self.n, w_init=self.output_w_init, b_init=self.b_init)(x)
        )

        return distrax.Normal(m_logits, jnp.exp(m_std_logs))


class ContinuousCritic(hk.Module):
    def __init__(self, config: dict, name: Optional[str] = None):
        super().__init__(name)

        self.encoder_w_init = hk.initializers.Orthogonal(np.sqrt(2))
        self.output_w_init = hk.initializers.Orthogonal(1.0)
        self.b_init = hk.initializers.Constant(0.0)

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)

        x = jnn.relu(hk.Linear(256, w_init=self.encoder_w_init, b_init=self.b_init)(x))
        x = jnn.relu(hk.Linear(256, w_init=self.encoder_w_init, b_init=self.b_init)(x))

        return hk.Linear(1, w_init=self.output_w_init, b_init=self.b_init)(x)


class DoubleContinuousCritic(hk.Module):
    def __init__(self, config: dict, name: Optional[str] = None):
        super().__init__(name)

        self.q1 = ContinuousCritic(config, "critic_1")
        self.q2 = ContinuousCritic(config, "critic_2")

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        q1 = self.q1(observations, actions)
        q2 = self.q2(observations, actions)
        return q1, q2


def get_continuous_networks(config: dict):
    key1, key2 = jrd.split(jrd.PRNGKey(config["seed"]), 2)
    observations = jnp.zeros((1,) + config["observation_shape"], jnp.float32)
    actions = jnp.zeros((1, config["n_actions"]), jnp.float32)

    @hk.transform
    def actor_transformed(observations):
        return ContinuousActor(config, "actor")(observations)

    actor_fwd = actor_transformed.apply
    actor_params = actor_transformed.init(key1, observations)

    @hk.transform
    def critic_transformed(observations, actions):
        return DoubleContinuousCritic(config, "critics")(observations, actions)

    critic_fwd = critic_transformed.apply
    critic_params = critic_transformed.init(key2, observations, actions)

    fwd = [actor_fwd, critic_fwd]
    params = [actor_params, critic_params]

    return fwd, params
