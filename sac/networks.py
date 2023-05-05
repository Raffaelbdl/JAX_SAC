from typing import Optional

import distrax
import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrd


class ContinuousActor(hk.Module):
    def __init__(self, config: dict, name: Optional[str] = None):
        super().__init__(name)

        self.n = config["n_actions"]

        self.log_std_max = 2
        self.log_std_min = -5

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = observations

        x = jnn.relu(hk.Linear(256)(x))
        x = jnn.relu(hk.Linear(256)(x))

        loc = hk.Linear(self.n, name="loc")(x)
        log_std = jnn.tanh(hk.Linear(self.n, name="log_std")(x))

        log_std = 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        log_std += self.log_std_min

        return distrax.Normal(loc, jnp.exp(log_std))


class ContinuousCritic(hk.Module):
    def __init__(self, config: dict, name: Optional[str] = None):
        super().__init__(name)

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)

        x = jnn.relu(hk.Linear(256)(x))
        x = jnn.relu(hk.Linear(256)(x))

        return hk.Linear(1)(x)


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
    key1, key2, key3 = jrd.split(jrd.PRNGKey(config["seed"]), 3)
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

    @hk.without_apply_rng
    @hk.transform
    def log_alpha_transformed():
        return hk.get_parameter("alpha", (), init=jnp.zeros)

    log_alpha_fwd = log_alpha_transformed.apply
    log_alpha_params = log_alpha_transformed.init(key3)

    fwd = [actor_fwd, critic_fwd, log_alpha_fwd]
    params = [actor_params, critic_params, log_alpha_params]

    return fwd, params
