"""Code taken from SB3, in order to not have useless imports"""

from typing import Dict

from gymnasium.spaces import Space
import numpy as np


class ReplayBuffer:
    """Buffer for off-policy algorithms"""

    def __init__(
        self,
        seed: int,
        buffer_size: int,
        obs_space: Space,
        action_space: Space,
        n_envs: int = 1,
    ) -> None:
        self.rng = np.random.default_rng(seed=seed)

        self.n_envs = n_envs
        self.buffer_size = max(buffer_size // self.n_envs, 1)

        self.obs_shape = (self.n_envs,) + obs_space.shape
        self.action_shape = (self.n_envs,) + action_space.shape

        self.observations = np.zeros(
            (self.buffer_size,) + self.obs_shape, obs_space.dtype
        )
        self.actions = np.zeros(
            (self.buffer_size,) + self.action_shape, action_space.dtype
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), np.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_obs: np.ndarray,
    ) -> None:
        obs = obs.reshape(self.obs_shape)
        action = action.reshape(self.action_shape)
        reward = reward.reshape((self.n_envs,))
        done = done.reshape((self.n_envs,))
        next_obs = obs.reshape(self.obs_shape)

        self.observations[self.pos] = obs.copy()
        self.actions[self.pos] = action.copy()
        self.rewards[self.pos] = reward.copy()
        self.dones[self.pos] = done.copy()
        self.observations[(self.pos + 1) % self.buffer_size] = next_obs.copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        if self.full:
            inds = self.rng.integers(1, self.buffer_size, size=batch_size)
            inds = (inds + self.pos) % self.buffer_size
        else:
            inds = self.rng.integers(0, self.pos, size=batch_size)
        return self._get_samples(inds)

    def _get_samples(self, inds: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[inds],
            "actions": self.actions[inds],
            "rewards": self.rewards[inds],
            "dones": self.dones[inds],
            "next_observations": self.observations[(inds + 1) % self.buffer_size],
        }
