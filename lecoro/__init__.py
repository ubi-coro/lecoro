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
"""
This file contains lists of available environments, dataset and algo to reflect the current state of LeRobot library.
We do not want to import all the dependencies, but instead we keep it lightweight to ensure fast access to these variables.

Example:
    ```python
        import lecoro
        print(lecoro.available_envs)
        print(lecoro.available_tasks_per_env)
        print(lecoro.available_datasets)
        print(lecoro.available_datasets_per_env)
        print(lecoro.available_real_world_datasets)
        print(lecoro.available_policies)
        print(lecoro.available_policies_per_env)
        print(lecoro.available_robots)
        print(lecoro.available_cameras)
        print(lecoro.available_motors)
    ```

When implementing a new dataset loadable with LeRobotDataset follow these steps:
- Update `available_datasets_per_env` in `lecoro/__init__.py`

When implementing a new environment (e.g. `gym_aloha`), follow these steps:
- Update `available_tasks_per_env` and `available_datasets_per_env` in `lecoro/__init__.py`

When implementing a new policy class (e.g. `DiffusionPolicy`) follow these steps:
- Update `available_policies` and `available_policies_per_env`, in `lecoro/__init__.py`
- Set the required `name` class attribute.
- Update variables in `tests/test_available.py` by importing your new Policy class
"""

import itertools

from lecoro.__version__ import __version__  # noqa: F401

# TODO(rcadene): Improve algo and envs. As of now, an item in `available_policies`
# refers to a yaml file AND a modeling name. Same for `available_envs` which refers to
# a yaml file AND a environment name. The difference should be more obvious.
available_tasks_per_env = {
    "aloha": [
        "AlohaInsertion-v0",
        "AlohaTransferCube-v0",
    ],
    "pusht": ["PushT-v0"],
    "xarm": ["XarmLift-v0"],
    "dora_aloha_real": ["DoraAloha-v0", "DoraKoch-v0", "DoraReachy2-v0"],
}
available_envs = list(available_tasks_per_env.keys())

available_datasets_per_env = {
    "aloha": [
        "lecoro/aloha_sim_insertion_human",
        "lecoro/aloha_sim_insertion_scripted",
        "lecoro/aloha_sim_transfer_cube_human",
        "lecoro/aloha_sim_transfer_cube_scripted",
        "lecoro/aloha_sim_insertion_human_image",
        "lecoro/aloha_sim_insertion_scripted_image",
        "lecoro/aloha_sim_transfer_cube_human_image",
        "lecoro/aloha_sim_transfer_cube_scripted_image",
    ],
    # TODO(alexander-soare): Add "lecoro/pusht_keypoints". Right now we can't because this is too tightly
    # coupled with tests.
    "pusht": ["lecoro/pusht", "lecoro/pusht_image"],
    "xarm": [
        "lecoro/xarm_lift_medium",
        "lecoro/xarm_lift_medium_replay",
        "lecoro/xarm_push_medium",
        "lecoro/xarm_push_medium_replay",
        "lecoro/xarm_lift_medium_image",
        "lecoro/xarm_lift_medium_replay_image",
        "lecoro/xarm_push_medium_image",
        "lecoro/xarm_push_medium_replay_image",
    ],
    "dora_aloha_real": [
        "lecoro/aloha_static_battery",
        "lecoro/aloha_static_candy",
        "lecoro/aloha_static_coffee",
        "lecoro/aloha_static_coffee_new",
        "lecoro/aloha_static_cups_open",
        "lecoro/aloha_static_fork_pick_up",
        "lecoro/aloha_static_pingpong_test",
        "lecoro/aloha_static_pro_pencil",
        "lecoro/aloha_static_screw_driver",
        "lecoro/aloha_static_tape",
        "lecoro/aloha_static_thread_velcro",
        "lecoro/aloha_static_towel",
        "lecoro/aloha_static_vinh_cup",
        "lecoro/aloha_static_vinh_cup_left",
        "lecoro/aloha_static_ziploc_slide",
    ],
}

available_real_world_datasets = [
    "lecoro/aloha_mobile_cabinet",
    "lecoro/aloha_mobile_chair",
    "lecoro/aloha_mobile_elevator",
    "lecoro/aloha_mobile_shrimp",
    "lecoro/aloha_mobile_wash_pan",
    "lecoro/aloha_mobile_wipe_wine",
    "lecoro/aloha_static_battery",
    "lecoro/aloha_static_candy",
    "lecoro/aloha_static_coffee",
    "lecoro/aloha_static_coffee_new",
    "lecoro/aloha_static_cups_open",
    "lecoro/aloha_static_fork_pick_up",
    "lecoro/aloha_static_pingpong_test",
    "lecoro/aloha_static_pro_pencil",
    "lecoro/aloha_static_screw_driver",
    "lecoro/aloha_static_tape",
    "lecoro/aloha_static_thread_velcro",
    "lecoro/aloha_static_towel",
    "lecoro/aloha_static_vinh_cup",
    "lecoro/aloha_static_vinh_cup_left",
    "lecoro/aloha_static_ziploc_slide",
    "lecoro/umi_cup_in_the_wild",
    "lecoro/unitreeh1_fold_clothes",
    "lecoro/unitreeh1_rearrange_objects",
    "lecoro/unitreeh1_two_robot_greeting",
    "lecoro/unitreeh1_warehouse",
    "lecoro/nyu_rot_dataset",
    "lecoro/utokyo_saytap",
    "lecoro/imperialcollege_sawyer_wrist_cam",
    "lecoro/utokyo_xarm_bimanual",
    "lecoro/tokyo_u_lsmo",
    "lecoro/utokyo_pr2_opening_fridge",
    "lecoro/cmu_franka_exploration_dataset",
    "lecoro/cmu_stretch",
    "lecoro/asu_table_top",
    "lecoro/utokyo_pr2_tabletop_manipulation",
    "lecoro/utokyo_xarm_pick_and_place",
    "lecoro/ucsd_kitchen_dataset",
    "lecoro/austin_buds_dataset",
    "lecoro/dlr_sara_grid_clamp",
    "lecoro/conq_hose_manipulation",
    "lecoro/columbia_cairlab_pusht_real",
    "lecoro/dlr_sara_pour",
    "lecoro/dlr_edan_shared_control",
    "lecoro/ucsd_pick_and_place_dataset",
    "lecoro/berkeley_cable_routing",
    "lecoro/nyu_franka_play_dataset",
    "lecoro/austin_sirius_dataset",
    "lecoro/cmu_play_fusion",
    "lecoro/berkeley_gnm_sac_son",
    "lecoro/nyu_door_opening_surprising_effectiveness",
    "lecoro/berkeley_fanuc_manipulation",
    "lecoro/jaco_play",
    "lecoro/viola",
    "lecoro/kaist_nonprehensile",
    "lecoro/berkeley_mvp",
    "lecoro/uiuc_d3field",
    "lecoro/berkeley_gnm_recon",
    "lecoro/austin_sailor_dataset",
    "lecoro/utaustin_mutex",
    "lecoro/roboturk",
    "lecoro/stanford_hydra_dataset",
    "lecoro/berkeley_autolab_ur5",
    "lecoro/stanford_robocook",
    "lecoro/toto",
    "lecoro/fmb",
    "lecoro/droid_100",
    "lecoro/berkeley_rpt",
    "lecoro/stanford_kuka_multimodal_dataset",
    "lecoro/iamlab_cmu_pickup_insert",
    "lecoro/taco_play",
    "lecoro/berkeley_gnm_cory_hall",
    "lecoro/usc_cloth_sim",
]

available_datasets = sorted(
    set(itertools.chain(*available_datasets_per_env.values(), available_real_world_datasets))
)

# lists all available algo from `lecoro/common/algo`
available_policies = [
    "act",
    "diffusion",
    "tdmpc",
    "vqbet",
]

# lists all available robots from `lecoro/common/robot_devices/robots`
available_robots = [
    "koch",
    "koch_bimanual",
    "aloha",
    "so100",
    "moss",
]

# lists all available cameras from `lecoro/common/robot_devices/cameras`
available_cameras = [
    "opencv",
    "intelrealsense",
]

# lists all available motors from `lecoro/common/robot_devices/motors`
available_motors = [
    "dynamixel",
    "feetech",
]

# keys and values refer to yaml files
available_policies_per_env = {
    "aloha": ["act"],
    "pusht": ["diffusion", "vqbet"],
    "xarm": ["tdmpc"],
    "koch_real": ["act_koch_real"],
    "aloha_real": ["act_aloha_real"],
    "dora_aloha_real": ["act_aloha_real"],
}

env_task_pairs = [(env, task) for env, tasks in available_tasks_per_env.items() for task in tasks]
env_dataset_pairs = [
    (env, dataset) for env, datasets in available_datasets_per_env.items() for dataset in datasets
]
env_dataset_policy_triplets = [
    (env, dataset, policy)
    for env, datasets in available_datasets_per_env.items()
    for dataset in datasets
    for policy in available_policies_per_env[env]
]
