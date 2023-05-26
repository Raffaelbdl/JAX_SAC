from collections import deque

import gymnasium as gym
import numpy as np

import wandb
from absl import app, flags

flags.DEFINE_string(
    "config_path", "./configs/sac_mujoco_config.yaml", "Path to the config"
)
flags.DEFINE_boolean("use_wandb", False, "Boolean to use wandb")
flags.DEFINE_string("entity", "entity", "Name of the wandb entity")
flags.DEFINE_string("project", "project", "Name of the wandb project")

FLAGS = flags.FLAGS

from rltools.buffer import ReplayBuffer
from rltools.config import get_config
from rltools.saver import Pickler
from rltools.update import avg

from sac.agent import SAC
from sac.networks.atari_networks import get_atari_networks


def train(config: dict):
    env = gym.make("HalfCheetah-v4")
    config["action_space"] = env.action_space
    config["observation_space"] = env.observation_space

    if FLAGS.use_wandb:
        wandb.init(entity=FLAGS.entity, project=FLAGS.project, config=config)

    agent = SAC(config)
    buffer = ReplayBuffer(
        config["seed"],
        config["buffer_size"],
        config["observation_space"],
        config["action_space"],
    )

    logs = {
        "tmp_episode_reward": 0.0,
        "avg_episode_reward": deque(maxlen=20),
    }

    name = config["name"]
    with Pickler(f"{name}_actor", True) as p:
        observation, _ = env.reset()
        for t in range(config["n_env_steps"]):
            if t < config["n_random_steps"]:
                action = env.action_space.sample()
            else:
                action = agent.get_action(np.expand_dims(observation, axis=0))[0][0]

            next_observation, reward, done, trunc, _ = env.step(action)

            logs["tmp_episode_reward"] += reward

            buffer.add(observation, action, reward, done, next_observation)
            observation = next_observation

            if done or trunc:
                logs["avg_episode_reward"].append(logs["tmp_episode_reward"])
                logs["tmp_episode_reward"] = 0.0

                observation, _ = env.reset()

                print(t, avg(logs["avg_episode_reward"]))
                if t >= config["n_random_steps"]:
                    if FLAGS.use_wandb:
                        wandb.log(
                            {
                                "step": t,
                                "q1_loss": logs["q1_loss"],
                                "q1_loss": logs["q1_loss"],
                                "policy_loss": logs["policy_loss"],
                                "alpha_loss": logs["alpha_loss"],
                                "log_alpha": logs["log_alpha"],
                                "episode_return": avg(logs["avg_episode_reward"]),
                            }
                        )

            if t >= config["n_random_steps"]:
                logs.update(agent.improve(buffer))
                p.update(agent.policy_params)


def main(_):
    config = get_config(FLAGS.config_path)
    train(config)


if __name__ == "__main__":
    app.run(main)
