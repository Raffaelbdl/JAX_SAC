"""One file implementation of SAC"""
from copy import copy

import distrax as dx
import haiku as hk
import jax
import jax.lax as jlax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import optax

from rl_tools.update import update


class Actor(hk.Module):
    def __init__(self, config: dict, name: str | None = None):
        super().__init__(name)
        self.n_actions = np.prod(np.array(config["action_space"].shape))
        self.log_std_min = config["log_std_min"]
        self.log_std_max = config["log_std_max"]

    def __call__(self, observations: np.ndarray):
        x = jnn.relu(hk.Linear(256)(observations))
        x = jnn.relu(hk.Linear(256)(x))

        locs = hk.Linear(self.n_actions)(x)

        log_stds = jnn.tanh(hk.Linear(self.n_actions)(x))
        log_stds = 0.5 * (self.log_std_max - self.log_std_min) * (log_stds + 1)
        log_stds += self.log_std_min

        return self.get_action(locs, log_stds)

    def get_action(self, locs, log_stds):
        stds = jnp.exp(log_stds)
        normal_dist = dx.Normal(locs, stds)
        samples, log_probs = normal_dist.sample_and_log_prob(seed=hk.next_rng_key())

        actions = jnn.tanh(samples)

        log_probs -= jnp.log(1 - jnp.square(actions) + 1e-6)
        log_probs = jnp.sum(log_probs, axis=-1, keepdims=True)

        return actions, log_probs


class QNetwork(hk.Module):
    def __init__(self, name: str | None = None):
        super().__init__(name)

    def __call__(self, observations: np.ndarray, actions: np.ndarray):
        x = jnp.concatenate([observations, actions], axis=-1)
        x = jnn.relu(hk.Linear(256)(x))
        x = jnn.relu(hk.Linear(256)(x))
        return hk.Linear(1)(x)


def get_continuous_networks(config: dict, key: jrd.PRNGKeyArray):
    @hk.transform
    def actor_transformed(observations: np.ndarray):
        return Actor(config, "actor")(observations)

    @hk.transform
    def critic_transformed(observations: np.ndarray, actions: np.ndarray):
        q1 = QNetwork("q1")(observations, actions)
        q2 = QNetwork("q2")(observations, actions)
        return q1, q2

    @hk.transform
    def alpha_transformed():
        return hk.get_parameter(
            "alpha", (), jnp.float32, hk.initializers.Constant(-3.5)
        )

    key1, key2, key3 = jrd.split(key, 3)
    dummy_observations = np.array(config["observation_space"].sample())[None, ...]
    dummy_actions = np.array(config["action_space"].sample())[None, ...]

    actor_params = actor_transformed.init(key1, dummy_observations)
    actor_fwd = actor_transformed.apply

    critic_params = critic_transformed.init(key2, dummy_observations, dummy_actions)
    critic_fwd = critic_transformed.apply

    alpha_params = alpha_transformed.init(key3)
    alpha_fwd = alpha_transformed.apply

    fwd = [actor_fwd, critic_fwd, alpha_fwd]
    params = [actor_params, critic_params, alpha_params]
    return fwd, params


def get_loss_fn(config, actor_fwd, critic_fwd, alpha_fwd):
    def critic_loss_fn(params, key, batch):
        rewards = jnp.expand_dims(batch["rewards"], axis=-1)
        dones = jnp.expand_dims(batch["dones"], axis=-1)
        next_observations = batch["next_observations"]

        next_actions, next_log_probs = actor_fwd(
            batch["actor_params"], key, next_observations
        )

        next_q1, next_q2 = critic_fwd(
            batch["target_critic_params"], None, next_observations, next_actions
        )
        next_q_min = jnp.minimum(next_q1, next_q2)

        discounts = config["gamma"] * (1 - dones)
        alpha = jnp.exp(alpha_fwd(batch["alpha_params"], None))

        targets = rewards + discounts * (next_q_min - alpha * next_log_probs)

        q1, q2 = critic_fwd(params, None, batch["observations"], batch["actions"])
        q1_loss = jnp.mean(jnp.square(q1 - targets))
        q2_loss = jnp.mean(jnp.square(q2 - targets))

        return q1_loss + q2_loss, {"q1_loss": q1_loss, "q2_loss": q2_loss}

    def actor_loss_fn(params, key, batch):
        observations = batch["observations"]
        actions, log_probs = actor_fwd(params, key, observations)

        # stop gradients just in case
        q1, q2 = critic_fwd(batch["critic_params"], None, observations, actions)
        min_q = jnp.minimum(q1, q2)
        log_alpha = alpha_fwd(batch["alpha_params"], None)
        alpha = jnp.exp(log_alpha)

        actor_loss = jnp.mean(alpha * log_probs - min_q)

        return actor_loss, {"actor_loss": actor_loss}

    def alpha_loss_fn(params, key, batch):
        target_entropy = -np.prod(np.array(config["action_space"].shape))

        _, log_probs = actor_fwd(batch["actor_params"], key, batch["observations"])
        log_probs = log_probs[..., 0]

        log_alpha = alpha_fwd(params, None)
        alpha_loss = jnp.mean(-log_alpha * (log_probs + target_entropy))

        return alpha_loss, {"log_alpha": log_alpha, "alpha_loss": alpha_loss}

    return actor_loss_fn, critic_loss_fn, alpha_loss_fn


def prepare_data_fn(batch):
    return {
        "observations": batch["observations"],
        "actions": batch["actions"],
        "rewards": batch["rewards"],
        "dones": batch["dones"],
        "next_observations": batch["next_observations"],
    }


class SAC:
    def __init__(self, config: dict) -> None:
        self.key = jrd.PRNGKey(config["seed"])

        fwd, params = get_continuous_networks(config, self._next_rng_key())
        self.actor_fwd, self.critic_fwd, self.alpha_fwd = fwd
        self.actor_params, self.critic_params, self.alpha_params = params
        self.target_critic_params = copy(self.critic_params)

        self.actor_loss_fn, self.critic_loss_fn, self.alpha_loss_fn = get_loss_fn(
            config, self.actor_fwd, self.critic_fwd, self.alpha_fwd
        )

        self.batch_size = config["batch_size"]
        self.action_space = config["action_space"]
        self.tau = config["tau"]

        self.init_optimizers()

    def get_action(self, observation):
        actions, _ = self.actor_fwd(
            self.actor_params, self._next_rng_key(), np.expand_dims(observation, axis=0)
        )

        return jnp.clip(actions, self.action_space.low, self.action_space.high)[0]

    def improve(self, buffer):
        data = prepare_data_fn(buffer.sample(self.batch_size))

        data["actor_params"] = self.actor_params
        data["alpha_params"] = self.alpha_params
        data["target_critic_params"] = self.target_critic_params
        self.critic_params, self.critic_opt_state, (critic_loss, _) = update(
            self.critic_params,
            self._next_rng_key(),
            data,
            self.critic_loss_fn,
            self.critic_optimizer,
            self.critic_opt_state,
        )

        data["critic_params"] = self.critic_params
        self.actor_params, self.actor_opt_state, (actor_loss, loss_dict) = update(
            self.actor_params,
            self._next_rng_key(),
            data,
            self.actor_loss_fn,
            self.actor_optimizer,
            self.actor_opt_state,
        )

        data["actor_params"] = self.actor_params
        self.alpha_params, self.alpha_opt_state, (alpha_loss, _) = update(
            self.alpha_params,
            self._next_rng_key(),
            data,
            self.alpha_loss_fn,
            self.alpha_optimizer,
            self.alpha_opt_state,
        )

        self.update_target_network()

        return actor_loss, critic_loss, alpha_loss

    def update_target_network(self) -> None:
        self.target_critic_params = jax.tree_util.tree_map(
            lambda p, t: self.tau * p + (1 - self.tau) * t,
            self.critic_params,
            self.target_critic_params,
        )

    def init_optimizers(self):
        self.actor_optimizer = optax.adam(1e-3)
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)
        self.critic_optimizer = optax.adam(1e-3)
        self.critic_opt_state = self.critic_optimizer.init(self.critic_params)
        self.alpha_optimizer = optax.adam(1e-3)
        self.alpha_opt_state = self.alpha_optimizer.init(self.alpha_params)

    def _next_rng_key(self):
        self.key, _key = jrd.split(self.key)
        return _key
