from collections import deque

import gymnasium as gym

from rl_tools.buffer import ReplayBuffer
from rl_tools.config import get_config
from rl_tools.saver import Pickler

from sac_one_file import SAC


def train_sac(config_path: str, **kwargs):
    config = get_config(config_path, kwargs)

    env = gym.make("HalfCheetah-v2")
    config["observation_space"] = env.observation_space
    config["action_space"] = env.action_space

    agent = SAC(config)
    buffer = ReplayBuffer(
        config["seed"],
        config["buffer_size"],
        env.observation_space,
        env.action_space,
    )

    logs = {
        "steps": 0,
        "episodes": 0,
        "n_updates": 0,
        "episode_return": 0,
        "tmp_episode_return": 0,
    }
    avg_reward_deque = deque(maxlen=20)
    actor_loss = 0.0
    critic_loss = 0.0
    alpha_loss = 0.0

    observation, _ = env.reset()
    n_iterations = config["n_env_steps"]

    with Pickler(config["name"] + "_actor", True) as pickler:
        for i in range(n_iterations):
            if logs["steps"] < config["start_training_step"]:
                action = env.action_space.sample()
            else:
                action = agent.get_action(observation)

            next_observation, reward, done, trunc, _ = env.step(action)
            logs["tmp_episode_return"] += reward

            buffer.add(observation, action, reward, done, next_observation)
            observation = next_observation

            if done or trunc:
                avg_reward_deque.append(logs["tmp_episode_return"])
                logs["tmp_episode_return"] = 0.0
                logs["episode_return"] = sum(avg_reward_deque) / len(avg_reward_deque)
                print(
                    logs["steps"],
                    logs["episode_return"],
                    f"{actor_loss:.2f}",
                    f"{critic_loss:.2f}",
                    f"{alpha_loss:.2f}",
                )

                observation, _ = env.reset()

            logs["steps"] += 1

            if logs["steps"] > config["start_training_step"]:
                actor_loss, critic_loss, alpha_loss = agent.improve(buffer)
                pickler.update(agent.actor_params)


if __name__ == "__main__":
    train_sac("./config.yaml")
