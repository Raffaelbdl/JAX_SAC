from functools import partial

import chex
import distrax as dx
from einops import rearrange
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np

from rl_tools.agent import Agent
from rl_tools.update import update

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
        self.actor_fwd, self.critic_fwd = fwd
        self.params = hk.data_structures.merge(*params)

        self.loss_fn = get_sac_loss_fn(config, self.actor_fwd, self.critic_fwd)
        self.prepare_data_fn = get_prepare_date_fn(
            config, self.actor_fwd, self.critic_fwd
        )

    def get_action(self, observations, params=None):
        params = params if params is not None else self.params

        dists = jax.jit(self.actor_fwd)(params, None, observations)
        actions = dists.sample(seed=self._next_rng_key())

        # TODO need to enforce action bound
        # see https://docs.cleanrl.dev/rl-algorithms/sac/#implementation-details

        return actions

    def improve(self, buffer):
        self.init_metrics()

        for i in range(self.n_training_steps):
            batch = self.prepare_data_fn(
                self.buffer.sample(self.batch_size), self.params, self._next_rng_key()
            )

            self.params, self.opt_state, (total_loss, loss_dict) = update(
                self.params,
                self._next_rng_key(),
                batch,
                self.loss_fn,
                self.optimizer,
                self.opt_state,
            )

        logs = {}
        return logs

    def _next_rng_key(self):
        self.key, _key = jrd.split(self.key)
        return _key


def get_prepare_date_fn(config: dict, actor_fwd, critic_fwd):
    def prepare_data_fn(buffer, params, key):
        # buffer (T, N_e, S)
        observations = buffer["observations"]
        actions = buffer["actions"]
        rewards = buffer["rewards"]
        discounts = config["gamma"] * jnp.where(buffer["dones"], 0.0, 1.0)
        next_observations = buffer["next_observations"]

        next_actions, next_log_probs = actor_fwd(
            params, None, next_observations
        ).sample_and_log_prob(seed=key)
        next_q1, next_q2 = critic_fwd(params, None, next_observations, next_actions)

        def get_target(r_t, q1_t, q2_t, logp_t, discount_t):
            return sac_target(r_t, q1_t, q2_t, logp_t, discount_t, config["alpha"])

        targets = jax.vmap(get_target, in_axes=1, out_axes=1)(
            rewards, next_q1, next_q2, next_log_probs, discounts
        )

        data = {}
        data["observations"] = rearrange(observations, "t n s -> (t n) s")
        data["actions"] = rearrange(actions, "t n s -> (t n) s")
        data["rewards"] = rearrange(rewards, "t n -> (t n)")
        data["discounts"] = rearrange(discounts, "t n -> (t n)")
        data["next_observations"] = rearrange(next_observations, "t n s -> (t n) s")
        data["targets"] = rearrange(targets, "t n -> (t n)")

        return data

    return prepare_data_fn


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
