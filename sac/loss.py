import jax.numpy as jnp
import jax.random as jrd
import numpy as np

from sac.networks.action_and_log_prob import get_actions_log_probs


def get_continous_sac_loss_fn(config: dict, policy_fwd, qvalue_fwd, alpha_fwd):
    def policy_loss_fn(params, key, batch):
        key1, key2 = jrd.split(key, 2)

        observations = batch["observations"]  # T S
        actions, log_probs = get_actions_log_probs(
            key1, policy_fwd(params, key2, observations)
        )  # T S
        log_probs = jnp.squeeze(log_probs, axis=-1)

        q1, q2 = qvalue_fwd(batch["qvalue_params"], None, observations, actions)
        min_q = jnp.minimum(q1, q2)
        log_alpha = alpha_fwd(batch["alpha_params"], None)
        alpha = jnp.exp(log_alpha)

        policy_loss = jnp.mean(alpha * log_probs - min_q)

        return policy_loss, {"policy_loss": policy_loss}

    def qvalue_loss_fn(params, key, batch):
        key1, key2 = jrd.split(key, 2)

        rewards = jnp.expand_dims(batch["rewards"], axis=-1)
        dones = jnp.expand_dims(batch["dones"], axis=-1)
        next_observations = batch["next_observations"]

        next_actions, next_log_probs = get_actions_log_probs(
            key1, policy_fwd(batch["policy_params"], key2, next_observations)
        )
        next_log_probs = jnp.squeeze(next_log_probs, axis=-1)

        next_q1, next_q2 = qvalue_fwd(
            batch["target_qvalue_params"], None, next_observations, next_actions
        )
        next_q_min = jnp.minimum(next_q1, next_q2)

        discounts = config["gamma"] * (1 - dones)
        alpha = jnp.exp(alpha_fwd(batch["alpha_params"], None))

        targets = rewards + discounts * (next_q_min - alpha * next_log_probs)

        q1, q2 = qvalue_fwd(params, None, batch["observations"], batch["actions"])
        q1_loss = jnp.mean(jnp.square(q1 - targets))
        q2_loss = jnp.mean(jnp.square(q2 - targets))

        return q1_loss + q2_loss, {"q1_loss": q1_loss, "q2_loss": q2_loss}

    def alpha_loss_fn(params, key, batch):
        key1, key2 = jrd.split(key, 2)

        target_entropy = -np.prod(np.array(config["action_space"].shape))

        _, log_probs = get_actions_log_probs(
            key1, policy_fwd(batch["policy_params"], key2, batch["observations"])
        )
        log_probs = jnp.squeeze(log_probs, axis=-1)

        log_alpha = alpha_fwd(params, None)
        alpha_loss = jnp.mean(-log_alpha * (log_probs + target_entropy))

        return alpha_loss, {"log_alpha": log_alpha, "alpha_loss": alpha_loss}

    return policy_loss_fn, qvalue_loss_fn, alpha_loss_fn


def get_discrete_sac_loss_fn(config: dict, policy_fwd, qvalue_fwd, alpha_fwd):
    def policy_loss_fn(params, key, batch):
        observations = batch["observations"]  # T S
        dist = policy_fwd(params, key, observations)

        probs = dist.probs
        log_probs = jnp.log(probs + 1e-8)  # T Sa

        q1, q2 = qvalue_fwd(batch["qvalue_params"], None, observations)
        min_q = jnp.minimum(q1, q2)  # T Sa

        log_alpha = alpha_fwd(batch["alpha_params"], None)
        alpha = jnp.exp(log_alpha)  # ()

        policy_loss = jnp.mean(jnp.sum(probs * (alpha * log_probs - min_q), axis=-1))

        return policy_loss, {"policy_loss": policy_loss}

    def qvalue_loss_fn(params, key, batch):
        rewards = jnp.expand_dims(batch["rewards"], axis=-1)
        dones = jnp.expand_dims(batch["dones"], axis=-1)
        next_observations = batch["next_observations"]

        next_dist = policy_fwd(batch["policy_params"], key, next_observations)
        next_probs = next_dist.probs
        next_log_probs = jnp.log(next_probs + 1e-8)

        next_q1, next_q2 = qvalue_fwd(
            batch["target_qvalue_params"], None, next_observations
        )
        next_q_min = jnp.minimum(next_q1, next_q2)

        discounts = config["gamma"] * (1 - dones)
        alpha = jnp.exp(alpha_fwd(batch["alpha_params"], None))

        targets = rewards + discounts * (next_q_min - alpha * next_log_probs)

        q1, q2 = qvalue_fwd(params, None, batch["observations"])
        q1_loss = jnp.mean(jnp.square(q1 - targets))
        q2_loss = jnp.mean(jnp.square(q2 - targets))

        return q1_loss + q2_loss, {"q1_loss": q1_loss, "q2_loss": q2_loss}

    def alpha_loss_fn(params, key, batch):
        target_entropy = -np.prod(np.array(config["action_space"].shape))

        dist = policy_fwd(batch["policy_params"], key, batch["observations"])
        probs = dist.probs
        log_probs = jnp.log(probs + 1e-8)

        log_alpha = alpha_fwd(params, None)
        alpha_loss = jnp.mean(
            jnp.sum(probs * (-log_alpha * (log_probs + target_entropy)), axis=-1)
        )

        return alpha_loss, {"log_alpha": log_alpha, "alpha_loss": alpha_loss}

    return policy_loss_fn, qvalue_loss_fn, alpha_loss_fn


def get_sac_loss_fn(config: dict, policy_fwd, qvalue_fwd, alpha_fwd):
    if config["discrete"]:
        return get_discrete_sac_loss_fn(config, policy_fwd, qvalue_fwd, alpha_fwd)
    return get_continous_sac_loss_fn(config, policy_fwd, qvalue_fwd, alpha_fwd)


def prepare_data_fn(buffer_dict: dict[str, np.ndarray]):
    return {
        "observations": buffer_dict["observations"],
        "actions": buffer_dict["actions"],
        "rewards": buffer_dict["rewards"],
        "dones": buffer_dict["dones"],
        "next_observations": buffer_dict["next_observations"],
    }
