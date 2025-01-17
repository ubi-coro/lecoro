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
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn


def populate_queues(queues, batch):
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note: assumes that all parameters have the same device
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype.
    """
    return next(iter(module.parameters())).dtype


def action_dict_to_vector(
        action_dict: dict[str, np.ndarray],
        action_keys: Optional[Sequence[str]]=None) -> np.ndarray:
    if action_keys is None:
        action_keys = list(action_dict.keys())
    actions = [action_dict[k] for k in action_keys]

    action_vec = np.concatenate(actions, axis=-1)
    return action_vec


def vector_to_action_dict(
        action: np.ndarray,
        action_shapes: dict[str, tuple[int]],
        action_keys: Sequence[str]) -> dict[str, np.ndarray]:
    action_dict = dict()
    start_idx = 0
    for key in action_keys:
        this_act_shape = action_shapes[key]
        this_act_dim = np.prod(this_act_shape)
        end_idx = start_idx + this_act_dim
        action_dict[key] = action[...,start_idx:end_idx].reshape(
            action.shape[:-1]+this_act_shape)
        start_idx = end_idx
    return action_dict
