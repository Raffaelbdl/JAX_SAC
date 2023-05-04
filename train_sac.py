from collections import deque
import pickle

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import wandb

from rl_tools.buffer import SimpleOnPolicyBuffer
from rl_tools.saver import Pickler

from envs.half_cheetah_envpool import EnvpoolHalfCheetah

# from ppo.modular_ppo import PPO
from sac.sac import SAC
from sac.read_config import get_config
from buffer import ReplayBuffer


def train_sac(config_path: str, **kwargs):
    config = get_config(config_path, kwargs)

    envs = EnvpoolHalfCheetah(
        config["seed"], n_envs=config["n_envs"], multi_agent=False
    )
    config["observation_shape"] = envs.observation_space.shape
    config["n_actions"] = envs.action_space.shape[0]
    config["n_agents"] = 1

    agent = SAC(config)
    buffer = ReplayBuffer(
        config["seed"],
        config["buffer_size"],
        envs.observation_space,
        envs.action_space,
        config["n_envs"],
    )

    logs = {
        "steps": 0,
        "episodes": 0,
        "n_updates": 0,
        "episode_return": 0,
        "tmp_episode_return": np.zeros((envs.n_envs, 1)),
    }
    avg_reward_deque = deque(maxlen=20)
    actor_loss = 0.0
    critic_loss = 0.0

    observations, _ = envs.reset()
    n_iterations = config["n_env_steps"] // config["n_envs"]
    for i in range(n_iterations):
        if logs["steps"] < config["start_training_step"]:
            actions = np.array(
                [envs.action_space.sample() for _ in range(config["n_envs"])]
            )
            actions = np.expand_dims(actions, axis=1)
        else:
            actions = agent.get_action(observations)

        next_observations, rewards, dones, truncs, _ = envs.step(actions)

        logs["tmp_episode_return"] += rewards
        buffer.add(observations, actions, rewards, dones, next_observations)

        observations = next_observations

        done = np.any(dones, axis=1)
        trunc = np.any(truncs, axis=1)
        for i, (d, t) in enumerate(zip(done, trunc)):
            if d or t:
                avg_reward_deque.append(logs["tmp_episode_return"][i][0])
                logs["episode_return"] = sum(avg_reward_deque) / len(avg_reward_deque)
                logs["tmp_episode_return"][i] = 0.0
                print(logs["steps"], logs["episode_return"], actor_loss, critic_loss)

        logs["steps"] += config["n_envs"]

        if logs["steps"] > config["start_training_step"]:
            actor_loss = agent.improve_actor(buffer)
            critic_loss = agent.improve_critic(buffer)


if __name__ == "__main__":
    train_sac("./configs/sac_config.yaml")
