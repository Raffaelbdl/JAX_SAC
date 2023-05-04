from functools import partial

import jax.numpy as jnp
from sac.distributions import normal_and_tanh_sample_and_log_prob


def get_sac_loss_fn(config, actor_fwd, critic_fwd):
    def critic_loss_fn(params, key, batch):
        observations = batch["observations"]
        actions = batch["actions"]
        targets = batch["targets"]

        q1_values, q2_values = critic_fwd(params, None, observations, actions)
        q1_loss = jnp.mean(jnp.square(q1_values - targets))
        q2_loss = jnp.mean(jnp.square(q2_values - targets))

        return q1_loss + q2_loss, {"q1_loss": q1_loss, "q2_loss": q2_loss}

    def actor_loss_fn(params, key, batch):
        observations = batch["observations"]
        dists = actor_fwd(params, None, observations)
        actions, log_probs = normal_and_tanh_sample_and_log_prob(key, dists)

        q1, q2 = critic_fwd(batch["critic_params"], None, observations, actions)

        log_probs = jnp.sum(log_probs, axis=-1, keepdims=True)
        actor_loss = -jnp.mean(jnp.fmin(q1, q2) - config["alpha"] * log_probs)

        return actor_loss, {"actor_loss": actor_loss}

    return actor_loss_fn, critic_loss_fn
