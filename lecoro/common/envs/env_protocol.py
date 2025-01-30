from abc import ABC, abstractmethod
from typing import Dict

import gymnasium as gym

from lecoro.common.envs.config_gen import ManipulatorEnvConfig


class ManipulatorEnv(gym.Env, ABC):
    def __init__(self, config: ManipulatorEnvConfig, action_space, observation_space):
        super().__init__()
        self.config = config
        self.action_space = action_space
        self.observation_space = observation_space

    @property
    @abstractmethod
    def features(self):
        pass

    @property
    @abstractmethod
    def produces_videos(self):
        pass

    @property
    @abstractmethod
    def fps(self):
        pass

    @property
    @abstractmethod
    def is_connected(self):
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the environment to an initial state and return the initial observation.
        Must be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Perform an action in the environment and return the next state, reward, done flag, and additional info.

        Args:
            action: The action to be taken by the manipulator.

        Returns:
            tuple: A tuple containing:
                - next_state (np.array): Next state of the environment.
                - reward (float): Reward obtained by taking the action.
                - done (bool): Whether the episode has ended.
                - info (dict): Additional information.
        """
        pass

    @abstractmethod
    def teleop_step(self, record_data) -> Dict:
        """
        Perform an action in the environment and return the next state, reward, done flag, and additional info.

        Args:
            action: The action to be taken by the manipulator.

        Returns:
            Dict: A tuple containing:
                - next_state (np.array): Next state of the environment.
                - reward (float): Reward obtained by taking the action.
                - done (bool): Whether the episode has ended.
                - info (dict): Additional information.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up resources used by the environment.
        """
        pass

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    def run_calibration(self):
        pass

    def capture_observation(self):
        pass

    def send_action(self, action):
        pass
