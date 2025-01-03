from dataclasses import dataclass, field
from typing import Any

from lecoro.common.robot_devices.cameras.utils import Camera


@dataclass
class ManipulatorEnvConfig:
    robot_type: str
    fps: int
    wrapper: Any = None
    leader_arms: dict[str, Any] = field(default_factory=lambda: {})
    follower_arms: dict[str, Any] = field(default_factory=lambda: {})

    cameras: dict[str, Camera] = field(default_factory=lambda: {})
    display_images: bool = True

    # Optionally limit the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length
    # as the number of motors in your follower arms (assumes all follower arms have the same number of
    # motors).
    max_relative_target: list[float] | float | None = None
