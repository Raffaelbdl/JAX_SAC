"""Base of the SAC implementation"""
from copy import copy

import jax
import jax.random as jrd
import optax

from rltools.agent import Agent
from rltools.buffer import ReplayBuffer
from rltools.types import Array, PRNGKey
from rltools.update import avg, update

from sac.networks.linear_networks import get_linear_networks
from sac.loss import get_sac_loss_fn, prepare_data_fn
from sac.networks.action_and_log_prob import get_actions_log_probs


class SAC(Agent):
    def __init__(self, config: dict, get_networks_fn=get_linear_networks) -> None:
        key = jrd.PRNGKey(config["seed"])
        super().__init__(key)

        fwd, params = get_networks_fn(config)
        self.policy_fwd, self.qvalue_fwd, self.alpha_fwd = fwd
        self.policy_params, self.qvalue_params, self.alpha_params = params
        self.target_qvalue_params = copy(self.qvalue_params)

        loss_fn = get_sac_loss_fn(
            config, self.policy_fwd, self.qvalue_fwd, self.alpha_fwd
        )
        self.policy_loss_fn, self.qvalue_loss_fn, self.alpha_loss_fn = loss_fn
        self.policy_loss_fn = jax.jit(self.policy_loss_fn)
        self.qvalue_loss_fn = jax.jit(self.qvalue_loss_fn)
        self.alpha_loss_fn = jax.jit(self.alpha_loss_fn)
        self.prepare_data_fn = jax.jit(prepare_data_fn)

        self.n_updates = config["n_updates"]
        self.batch_size = config["batch_size"]
        self.tau = config["tau"]

        self.init_optimizer(config)

    def get_action(self, observation) -> tuple[Array, Array]:
        return get_actions_log_probs(
            self._next_rng_key(),
            self.policy_fwd(self.policy_params, self._next_rng_key(), observation),
        )

    def improve(self, buffer: ReplayBuffer):
        q1_loss = []
        q2_loss = []
        policy_loss = []
        alpha_loss = []
        log_alpha = []

        for i in range(self.n_updates):
            data = self.prepare_data_fn(buffer.sample(self.batch_size))

            data["policy_params"] = self.policy_params
            data["alpha_params"] = self.alpha_params
            data["target_qvalue_params"] = self.target_qvalue_params
            (
                self.qvalue_params,
                self.qvalue_opt_state,
                (_, qv_loss_dict),
            ) = update(
                self.qvalue_params,
                self._next_rng_key(),
                data,
                self.qvalue_loss_fn,
                self.qvalue_optimizer,
                self.qvalue_opt_state,
            )

            data["qvalue_params"] = self.qvalue_params
            (
                self.policy_params,
                self.policy_opt_state,
                (_, policy_loss_dict),
            ) = update(
                self.policy_params,
                self._next_rng_key(),
                data,
                self.policy_loss_fn,
                self.policy_optimizer,
                self.policy_opt_state,
            )

            data["policy_params"] = self.policy_params
            (
                self.alpha_params,
                self.alpha_opt_state,
                (_, alpha_loss_dict),
            ) = update(
                self.alpha_params,
                self._next_rng_key(),
                data,
                self.alpha_loss_fn,
                self.alpha_optimizer,
                self.alpha_opt_state,
            )

            self.update_target_network()

            q1_loss.append(qv_loss_dict["q1_loss"])
            q2_loss.append(qv_loss_dict["q2_loss"])
            policy_loss.append(policy_loss_dict["policy_loss"])
            alpha_loss.append(alpha_loss_dict["alpha_loss"])
            log_alpha.append(alpha_loss_dict["log_alpha"])

        return {
            "q1_loss": avg(q1_loss),
            "q2_loss": avg(q2_loss),
            "policy_loss": avg(policy_loss),
            "alpha_loss": avg(alpha_loss),
            "log_alpha": avg(log_alpha),
        }

    def update_target_network(self) -> None:
        self.target_qvalue_params = jax.tree_util.tree_map(
            lambda p, t: self.tau * p + (1 - self.tau) * t,
            self.qvalue_params,
            self.target_qvalue_params,
        )

    def init_optimizer(self, config: dict):
        self.policy_optimizer = optax.adam(config["policy_lr"])
        self.policy_opt_state = self.policy_optimizer.init(self.policy_params)

        self.qvalue_optimizer = optax.adam(config["qvalue_lr"])
        self.qvalue_opt_state = self.qvalue_optimizer.init(self.qvalue_params)

        self.alpha_optimizer = optax.adam(config["alpha_lr"])
        self.alpha_opt_state = self.alpha_optimizer.init(self.alpha_params)
