import hydra
from omegaconf import DictConfig

from lecoro.common.robot_devices.robots.utils import Robot


def make_robot(cfg: DictConfig) -> Robot:
    robot = hydra.utils.instantiate(cfg)
    return robot
