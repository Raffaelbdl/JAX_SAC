from functools import partial
from copy import copy
import chex
import distrax as dx
from einops import rearrange
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import optax

from rl_tools.agent import Agent
from rl_tools.distributions import normal_and_tanh_sample_and_log_prob
from rl_tools.update import update, apply_updates

from sac.loss import get_sac_loss_fn
from sac.networks import get_continuous_networks


class SAC(Agent):
    def __init__(
        self,
        config: dict,
        networks_factory=get_continuous_networks,
        loss_fn_factory=get_sac_loss_fn,
    ) -> None:
        self.seed = config["seed"]
        self.key = jrd.PRNGKey(self.seed)

        fwd, params = networks_factory(config)
        self.actor_fwd, self.critic_fwd, self.log_alpha_fwd = fwd
        self.actor_params, self.critic_params, self.log_alpha_params = params
        self.t_critic_params = copy(self.critic_params)

        self.actor_lr = config["actor_lr"]
        self.critic_lr = config["critic_lr"]

        self.tau = config["tau"]

        (
            self.actor_loss_fn,
            self.critic_loss_fn,
            self.log_alpha_loss_fn,
        ) = loss_fn_factory(config, self.actor_fwd, self.critic_fwd, self.log_alpha_fwd)

        self.actor_prepare_data_fn = prepare_actor_data_fn
        self.critic_prepare_data_fn = jax.jit(
            partial(
                prepare_critic_data_fn,
                config,
                self.actor_fwd,
                self.critic_fwd,
                self.log_alpha_fwd,
            )
        )

        self.n_env_steps = config["n_env_steps"]
        self.n_envs = config["n_envs"]
        self.n_actions = config["n_actions"]
        self.n_agents = config["n_agents"]

        self.n_actor_updates = config["n_actor_updates"]
        self.n_critic_updates = config["n_critic_updates"]
        self.batch_size = config["batch_size"]

        self.init_optimizer()

    def get_action(self, observations, params=None):
        params = params if params is not None else self.actor_params

        def _actions(params, key, observations):
            dists = self.actor_fwd(params, None, observations)
            actions, _ = normal_and_tanh_sample_and_log_prob(key, dists)
            return actions

        return jax.jit(_actions)(params, self._next_rng_key(), observations)

    def improve(self, buffer):
        actor_losses = []
        critic_losses = []
        alpha_losses = []

        n_updates = max([self.n_actor_updates, self.n_critic_updates])
        for i in range(n_updates):
            batch = self.critic_prepare_data_fn(
                self.actor_params,
                self.t_critic_params,
                self.log_alpha_params,
                self._next_rng_key(),
                buffer.sample(self.batch_size),
            )

            critic_losses.append(self.improve_critic(batch))
            actor_losses.append(self.improve_actor(batch))
            alpha_losses.append(self.improve_alpha(batch))
            self.update_target_network()

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)
        alpha_loss = sum(alpha_losses) / len(alpha_losses)

        return actor_loss, critic_loss, alpha_loss

    def improve_actor(self, batch):
        batch["critic_params"] = self.critic_params
        batch["log_alpha_params"] = self.log_alpha_params

        output = update(
            self.actor_params,
            self._next_rng_key(),
            batch,
            self.actor_loss_fn,
            self.actor_optimizer,
            self.actor_opt_state,
        )
        (self.actor_params, self.actor_opt_state, (loss, loss_dict)) = output

        return loss

    def improve_critic(self, batch):
        batch["log_alpha_params"] = self.log_alpha_params

        output = update(
            self.critic_params,
            self._next_rng_key(),
            batch,
            self.critic_loss_fn,
            self.critic_optimizer,
            self.critic_opt_state,
        )
        (self.critic_params, self.critic_opt_state, (loss, loss_dict)) = output

        return loss

    def improve_alpha(self, batch):
        batch["actor_params"] = self.actor_params

        output = update(
            self.log_alpha_params,
            self._next_rng_key(),
            batch,
            self.log_alpha_loss_fn,
            self.log_alpha_optimizer,
            self.log_alpha_opt_state,
        )
        (
            self.log_alpha_params,
            self.log_alpha_opt_state,
            (loss, loss_dict),
        ) = output

        return loss

    def init_optimizer(self) -> None:
        self.actor_optimizer = optax.adam(self.actor_lr)
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)

        self.critic_optimizer = optax.adam(self.critic_lr)
        self.critic_opt_state = self.critic_optimizer.init(self.critic_params)

        self.log_alpha_optimizer = optax.adam(self.critic_lr)
        self.log_alpha_opt_state = self.log_alpha_optimizer.init(self.log_alpha_params)

    def update_target_network(self) -> None:
        self.t_critic_params = jax.tree_util.tree_map(
            lambda p, t: self.tau * p + (1 - self.tau) * t,
            self.critic_params,
            self.t_critic_params,
        )

    def _next_rng_key(self):
        self.key, _key = jrd.split(self.key)
        return _key


def prepare_actor_data_fn(buffer):
    return {
        "observations": rearrange(buffer["observations"], "t n s -> (t n) s"),
        "actions": rearrange(buffer["actions"], "t n s -> (t n) s"),
    }


def prepare_critic_data_fn(
    config,
    actor_fwd,
    critic_fwd,
    log_alpha_fwd,
    actor_params,
    t_critic_params,
    log_alpha_params,
    key,
    buffer,
):
    next_observations = rearrange(buffer["next_observations"], "t n s -> (t n) s")

    next_actions, next_log_probs = actor_fwd(
        actor_params, None, next_observations
    ).sample_and_log_prob(seed=key)
    next_q1, next_q2 = critic_fwd(
        t_critic_params, None, next_observations, next_actions
    )

    next_q1, next_q2 = next_q1[..., 0], next_q2[..., 0]
    next_log_probs = jnp.sum(next_log_probs, axis=-1)

    rewards = rearrange(buffer["rewards"], "t n -> (t n)")
    dones = rearrange(buffer["dones"], "t n -> (t n)")
    discounts = config["gamma"] * (1.0 - dones)
    targets = sac_target(
        rewards,
        next_q1,
        next_q2,
        next_log_probs,
        discounts,
        log_alpha_fwd(log_alpha_params),
    )

    return {
        "observations": rearrange(buffer["observations"], "t n s -> (t n) s"),
        "actions": rearrange(buffer["actions"], "t n s -> (t n) s"),
        "targets": targets,
        "dones": dones,
        "discounts": discounts,
    }


def sac_target(r_t, q1_t, q2_t, logp_t, discount_t, alpha):
    """
    r_t: rewards collected in the destination state [1, T]
    q1_t: action value 1 in the destination state [1, T]
    q2_t: action value 2 in the destination state [1, T]
    pi_t: log_probs in the destination state [1, T]
    discount_t: discount associated with a transition [1, T]
    alpha: entropy coefficient
    """
    chex.assert_rank(
        [r_t, q1_t, q2_t, logp_t, discount_t, alpha], [1, 1, 1, 1, 1, {0, 1}]
    )
    chex.assert_type([r_t, q1_t, q2_t, logp_t, discount_t, alpha], float)
    chex.assert_equal_shape([r_t, q1_t, q2_t, logp_t])

    alpha = jnp.ones_like(r_t) * alpha
    return r_t + discount_t * (jnp.fmin(q1_t, q2_t) - alpha * logp_t)


def custom_update(
    params: hk.Params,
    key: jrd.PRNGKeyArray,
    batch: dict,
    loss_fn,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
):
    output, grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key, batch)
    params, opt_state = apply_updates(optimizer, params, opt_state, grads)
    return params, opt_state, output
