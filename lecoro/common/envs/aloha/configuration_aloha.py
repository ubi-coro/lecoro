from dataclasses import dataclass, field
from typing import List

from coro.common.config_gen.env.base import ManipulatorEnvConfig
from coro.common.devices.bota import Bota


@dataclass
class AlohaManipulatorConfig(ManipulatorEnvConfig):
    robot_type: str = "aloha"
    fps: int = 40
    display_images: bool = True

    moving_time: float = 0.1
    max_episode_length: int = 1000
    start_pos: List = field(default_factory= lambda: [0.0, 130.0, 130.0, 140.0, 140.0, 0.0, -10.0, 0.0, 90.0])
    rest_pos: List = field(default_factory= lambda: [0.0, 185.0, 185.0, 180.0, 180.0, 0.0, 20.0, 0.0, 5.0])

    # Optional use bota sensors to collect force-torque information. In the future, these should go into
    # the MotorsBus class to allow more tightly integrated control
    botas: dict[str, Bota] = field(default_factory=lambda: {})


