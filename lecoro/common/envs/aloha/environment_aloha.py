from pathlib import Path
import copy
from dataclasses import replace
import json

import numpy as np
from gymnasium.core import RenderFrame
from pynput import keyboard
import time
from typing import Dict

import torch
import gymnasium as gym

from coro.common.config_gen.env.aloha import AlohaManipulatorConfig
from coro.common.env.base import ManipulatorEnv
from coro.common.devices.robots.dynamixel import BaseManipulator
from coro.common.utils.action_utils import ensure_safe_goal_position
from coro.common.utils.file_utils import get_package_root
from coro.common.utils.video_utils import Displayer
from coro.common.utils.robot_utils import get_arm_id, RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError


class AlohaManipulatorEnv(ManipulatorEnv):
    def __init__(self,
                 config: AlohaManipulatorConfig | None = None,
                 calibration_dir: Path = ".cache/calibration/aloha_default",
                 **kwargs):
        if config is None:
            config = AlohaManipulatorConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)
        self.calibration_dir = get_package_root() / Path(calibration_dir)

        self.robot_type = self.config.robot_type
        self.leader_arms: Dict[str, BaseManipulator] = self.config.leader_arms
        self.follower_arms: Dict[str, BaseManipulator] = self.config.follower_arms

        self.cameras = self.config.cameras
        self.cam_axes = {}
        self.botas = self.config.botas

        self.start_pos = np.array(self.config.start_pos).astype(np.int32)
        self.rest_pos = np.array(self.config.rest_pos).astype(np.int32)
        self.curr_episode_length = 0
        self.last_step_ts = 0.0
        self.max_episode_length = self.config.max_episode_length
        self.max_relative_target = self.config.max_relative_target
        self.is_connected = False
        self.logs = {}

        # define action and obs spaces
        num_joints = sum([len(self.follower_arms[name].joint_names) for name in self.follower_arms])
        action_space = gym.spaces.Box(
            np.ones((num_joints,), dtype=np.float32) * -180.0,
            np.ones((num_joints,), dtype=np.float32) * 180.0,
        )

        obs_dict = {'observation.low_dim.qpos': copy.deepcopy(action_space)}
        if self.has_camera:
            for name in self.cameras:
                width = self.cameras[name].width if self.cameras[name].width is not None else 256
                height = self.cameras[name].height if self.cameras[name].height is not None else 256
                shape = (width, height, 3)
                obs_dict[f'observation.rgb.{name}'] = gym.spaces.Box(0, 255, shape)
        observation_space = gym.spaces.Dict(obs_dict)
        super().__init__(self.config, action_space=action_space, observation_space=observation_space)

        # todo: handle depth, wrenches

        # start flask server for displaying images
        if self.has_camera and self.config.display_images:
            self.displayer = Displayer()
            self.displayer.start_server()

        # start key listener for episode termination
        self.terminate = False
        def on_press(key):
            if key == keyboard.Key.esc:
                self.terminate = True
        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()

    def get_joint_names(self, arm: dict[str, BaseManipulator]) -> list:
        return [f"{arm}_{joint}" for arm, manipulator in arm.items() for joint in manipulator.joint_names]

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.rgb.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = self.get_joint_names(self.leader_arms)
        qpos_names = self.get_joint_names(self.leader_arms)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.low_dim.qpos": {
                "dtype": "float32",
                "shape": (len(qpos_names),),
                "names": qpos_names,
            },
        }

    @property
    def features(self):
        # todo: add bota features as well
        return {**self.motor_features, **self.camera_features}

    @property
    def fps(self):
        return self.config.fps

    @property
    def produces_videos(self):
        return self.has_camera

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def has_bota(self):
        return len(self.botas) > 0

    @property
    def num_botas(self):
        return len(self.botas)

    @property
    def available_arms(self):
        available_arms = []
        for name in self.follower_arms:
            arm_id = get_arm_id(name, "follower")
            available_arms.append(arm_id)
        for name in self.leader_arms:
            arm_id = get_arm_id(name, "leader")
            available_arms.append(arm_id)
        return available_arms

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras and not self.botas:
            raise ValueError(
                "ManipulatorRobot doesn't have any device to connect. See example of usage in docstring of the class."
            )

        # Connect the arms
        for name in self.follower_arms:
            print(f"Connecting {name} follower arm.")
            self.follower_arms[name].connect()
        for name in self.leader_arms:
            print(f"Connecting {name} leader arm.")
            self.leader_arms[name].connect()

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        self.toggle_torque(leader=False, follower=False)

        self.run_calibration()

        # Set robot preset (e.g. torque in leader gripper for Koch v1.1)
        for name in self.follower_arms:
            self.follower_arms[name].set_presets(robot_type='follower')
        for name in self.leader_arms:
            self.leader_arms[name].set_presets(robot_type='leader')

        # Enable torque on all motors of the follower arms
        for name in self.follower_arms:
            print(f"Activating torque on {name} follower arm.")
            self.follower_arms[name].torque_on()

        # Check both arms can be read
        #  if calibration errors we catch that later
        for name in self.follower_arms:
            self.follower_arms[name].get_joint_positions(apply_calibration=False)
        for name in self.leader_arms:
            self.leader_arms[name].get_joint_positions(apply_calibration=False)

        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        # Connect the botas
        for name in self.botas:
            self.botas[name].connect()

        self.is_connected = True

    def toggle_torque(self,
                      leader=True, leader_joints=None,
                      follower=True, follower_joints=None
                      ):
        for name in self.follower_arms:
            if follower:
                self.follower_arms[name].torque_on(joint_names=follower_joints)
            else:
                self.follower_arms[name].torque_off(joint_names=follower_joints)

        for name in self.leader_arms:
            if leader:
                self.leader_arms[name].torque_on(joint_names=leader_joints)
            else:
                self.leader_arms[name].torque_off(joint_names=leader_joints)

    @property
    def gripper_pos(self):
        return [self.leader_arms[name].get_single_joint_position("gripper") for name in self.leader_arms]

    def run_calibration(self):
        """After calibration all motors function in human interpretable ranges.
        Rotations are expressed in degrees in nominal range of [-180, 180],
        and linear motions (like gripper of Aloha) in nominal range of [0, 100].
        """

        def load_or_run_calibration_(name, arm, arm_type):
            arm_id = get_arm_id(name, arm_type)
            arm_calib_path = self.calibration_dir / f"{arm_id}.json"

            if arm_calib_path.exists():
                with open(arm_calib_path) as f:
                    calibration = json.load(f)
            else:
                print(f"Missing calibration file '{arm_calib_path}'")

                calibration = arm.run_arm_calibration(name, arm_type)

                print(f"Calibration is done! Saving calibration file '{arm_calib_path}'")
                arm_calib_path.parent.mkdir(parents=True, exist_ok=True)
                with open(arm_calib_path, "w") as f:
                    json.dump(calibration, f)

            return calibration

        for name, arm in self.follower_arms.items():
            calibration = load_or_run_calibration_(name, arm, "follower")
            arm.set_calibration(calibration)
        for name, arm in self.leader_arms.items():
            calibration = load_or_run_calibration_(name, arm, "leader")
            arm.set_calibration(calibration)

    def step(self, action):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        dt = time.perf_counter() - self.last_step_ts
        time.sleep(max([0, 1 / self.config.fps - dt]))
        self.last_step_ts = time.perf_counter()

        obs = self.capture_observation()
        obs['action'] = self.send_action(action)
        reward = self.compute_reward(obs)
        done = self.curr_episode_length >= self.max_episode_length or reward or self.terminate

        return obs, int(reward), done, False, {"succeed": reward}

    def reset(self):
        self.toggle_torque(leader=True, follower=True)
        self._go_to_start_pos()

        # todo: remove this until blocking works
        time.sleep(1.0)

        # torque leader gripper off
        for name in self.leader_arms:
            self.leader_arms[name].torque_off(joint_names=['gripper'])

        for name in self.follower_arms:
            self.follower_arms[name].set_trajectory_time(moving_time=self.config.moving_time)

        self.curr_episode_length = 0
        self.last_step_ts = time.perf_counter()
        obs = self.capture_observation()
        self.terminate = False
        return obs, {"succeed": False}

    def send_action(self, action):
        from_idx = 0
        to_idx = 0
        action_sent = []
        for name in self.follower_arms:

            # Get goal position of each follower arm by splitting the action vector
            to_idx += len(self.follower_arms[name].joint_names)
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.max_relative_target is not None:
                present_pos = self.follower_arms[name].get_joint_positions()
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.max_relative_target)

            # Save tensor to concat and return
            action_sent.append(goal_pos)

            # Send goal position to each follower
            goal_pos = goal_pos.numpy().astype(np.int32)
            self.follower_arms[name].set_joint_positions(goal_pos)
        return torch.cat(action_sent)

    def compute_reward(self, obs):
        return 0

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Prepare to assign the position of the leader to the follower
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].get_joint_positions()
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

        # sleep to match frequency
        dt = time.perf_counter() - self.last_step_ts
        time.sleep(max([0, 1 / self.config.fps - dt]))
        self.last_step_ts = time.perf_counter()

        # Send goal position to the follower
        follower_goal_pos = sum(leader_pos.values())
        follower_goal_pos = self.send_action(follower_goal_pos)

        # Early exit when recording data is not requested
        if not record_data:
            return

        frame = self.capture_observation()
        frame['action'] = follower_goal_pos

        return frame

    def print_logs(self):
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        for name in self.follower_arms:
            self.follower_arms[name].disconnect()

        for name in self.leader_arms:
            self.leader_arms[name].disconnect()

        for name in self.cameras:
            self.cameras[name].disconnect()

        for name in self.botas:
            self.botas[name].disconnect()

        self.is_connected = False

    def close(self):
        try:
            self._go_to_rest_pos()
        except RobotDeviceNotConnectedError:
            pass
        if getattr(self, "is_connected", True):
            self.disconnect()
        if hasattr(self, 'listener'):
            self.listener.stop()
        if hasattr(self, 'displayer'):
            self.displayer.stop_server()

    def _go_to_start_pos(self):
        for name in self.follower_arms:
            self.follower_arms[name].set_joint_positions(self.start_pos, moving_time=1.0)
        for name in self.leader_arms:
            self.leader_arms[name].set_joint_positions(self.start_pos, moving_time=1.0)

    def _go_to_rest_pos(self):
        for name in self.follower_arms:
            self.follower_arms[name].set_joint_positions(self.rest_pos, moving_time=1.0)

    def capture_observation(self):
        obs = {'observation.low_dim.qpos': self._get_qpos()}
        if self.has_camera:
            rgb = self._get_rgb()
            for name in rgb:
                obs[f'observation.rgb.{name}'] = rgb[name]
        if self.has_bota:
            obs['observation.low_dim.wrench'] = self._get_wrench()
        return obs

    def _get_qpos(self):
        # Create state by concatenating current follower position
        qpos = []
        for name in self.follower_arms:
            _qpos = self.follower_arms[name].get_joint_positions()
            qpos.append(torch.from_numpy(_qpos))

        if qpos:
            qpos = torch.cat(qpos)
        return qpos

    def _get_rgb(self):
        # get current frames
        display_images = {}
        images = {}
        for name in self.cameras:
            img = self.cameras[name].async_read()
            display_images[name] = img
            images[name] = torch.from_numpy(img)

        if display_images:
            self.displayer.update_images(display_images)

        # record frames
        return images

    def _get_wrench(self):
        # get current frames
        wrenches = []
        for name in self.botas:
            wrench = self.botas[name].async_read()
            wrenches.append(torch.from_numpy(wrench))

        # record frames
        return wrenches

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def __del__(self):
        self.close()


# todo: write aloha simulation

