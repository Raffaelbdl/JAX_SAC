from functools import partial

import jax.numpy as jnp


def sac_loss_fn(config, actor_fwd, critic_fwd, params, batch):
    def critic_loss_fn(params, batch):
        observations = batch["observations"]
        actions = batch["actions"]
        targets = batch["targets"]

        q1_values, q2_values = critic_fwd(params, None, observations, actions)
        q1_loss = jnp.mean(jnp.square(q1_values - targets))
        q2_loss = jnp.mean(jnp.square(q2_values - targets))

        return q1_loss + q2_loss, {"q1_loss": q1_loss, "q2_loss": q2_loss}

    def actor_loss_fn(params, batch):
        observations = batch["observations"]
        actions = batch["actions"]
        q1_values, q2_values = batch["q1_values"], batch["q2_values"]

        # TODO check if actions must be sampled again
        log_probs = actor_fwd(params, None, observations).log_prob(actions)
        actor_loss = jnp.mean(
            jnp.fmin(q1_values, q2_values) - config["alpha"] * log_probs
        )

        return actor_loss, {"actor_loss": actor_loss}

    actor_loss, actor_loss_dict = actor_loss_fn(params, batch)
    critic_loss, critic_loss_dict = critic_loss_fn(params, batch)

    total_loss = actor_loss + critic_loss
    loss_dict = actor_loss_dict
    loss_dict.update(critic_loss_dict)

    return total_loss, loss_dict


def get_sac_loss_fn(config: dict, actor_fwd, critic_fwd):
    return partial(sac_loss_fn, config, actor_fwd, critic_fwd)
