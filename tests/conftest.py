#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import traceback

import pytest
from serial import SerialException

from lecoro import available_cameras, available_motors, available_robots
from lecoro.common.utils.utils import init_hydra_config
from tests.utils import DEVICE, ROBOT_CONFIG_PATH_TEMPLATE, make_camera, make_motors_bus

# Import fixture modules as plugins
pytest_plugins = [
    "tests.fixtures.dataset_factories",
    "tests.fixtures.files",
    "tests.fixtures.hub",
]


def pytest_collection_finish():
    print(f"\nTesting with {DEVICE=}")


@pytest.fixture
def is_robot_available(robot_type):
    if robot_type not in available_robots:
        raise ValueError(
            f"The robot type '{robot_type}' is not valid. Expected one of these '{available_robots}"
        )

    try:
        from lecoro.common.robot_devices.robots.factory import make_robot

        config_path = ROBOT_CONFIG_PATH_TEMPLATE.format(robot=robot_type)
        robot_cfg = init_hydra_config(config_path)
        robot = make_robot(robot_cfg)
        robot.connect()
        del robot
        return True

    except Exception as e:
        print(f"\nA {robot_type} robot is not available.")

        if isinstance(e, ModuleNotFoundError):
            print(f"\nInstall module '{e.name}'")
        elif isinstance(e, SerialException):
            print("\nNo physical motors bus detected.")
        else:
            traceback.print_exc()

        return False


@pytest.fixture
def is_camera_available(camera_type):
    if camera_type not in available_cameras:
        raise ValueError(
            f"The camera type '{camera_type}' is not valid. Expected one of these '{available_cameras}"
        )

    try:
        camera = make_camera(camera_type)
        camera.connect()
        del camera
        return True

    except Exception as e:
        print(f"\nA {camera_type} camera is not available.")

        if isinstance(e, ModuleNotFoundError):
            print(f"\nInstall module '{e.name}'")
        elif isinstance(e, ValueError) and "camera_index" in e.args[0]:
            print("\nNo physical camera detected.")
        else:
            traceback.print_exc()

        return False


@pytest.fixture
def is_motor_available(motor_type):
    if motor_type not in available_motors:
        raise ValueError(
            f"The motor type '{motor_type}' is not valid. Expected one of these '{available_motors}"
        )

    try:
        motors_bus = make_motors_bus(motor_type)
        motors_bus.connect()
        del motors_bus
        return True

    except Exception as e:
        print(f"\nA {motor_type} motor is not available.")

        if isinstance(e, ModuleNotFoundError):
            print(f"\nInstall module '{e.name}'")
        elif isinstance(e, SerialException):
            print("\nNo physical motors bus detected.")
        else:
            traceback.print_exc()

        return False


@pytest.fixture
def patch_builtins_input(monkeypatch):
    def print_text(text=None):
        if text is not None:
            print(text)

    monkeypatch.setattr("builtins.input", print_text)
