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
"""A protocol that all algo should follow.

This provides a mechanism for type-hinting and isinstance checks without requiring the algo classes
subclass a base class.

The protocol structure, method signatures, and docstrings should be used by developers as a reference for
how to implement new algo.
"""

from typing import Protocol, runtime_checkable

from torch import Tensor
"""
Heavily inspired by robomimic's algorithm implementations.

This file contains base classes that other algorithm classes subclass.
Each algorithm file also implements a algorithm factory function that
takes in an algorithm config (`config.algo`) and returns the particular
Algo subclass that should be instantiated, along with any extra kwargs.
These factory functions are registered into a global dictionary with the
@register_algo_factory_func function decorator. This makes it easy for
@algo_factory to instantiate the correct `Algo` subclass.
"""
from contextlib import nullcontext
from collections import OrderedDict
import copy
import time
from typing import Callable

import torch
import torch.nn as nn
from lerobot.common.policies.normalize import Normalize, Unnormalize
from torch import Tensor
from torch.cuda.amp import GradScaler

from coro.common.config_gen import ObsConfig
from coro.common.utils.obs_utils import all_keys, get_normalization_mode, process_obs_dict, OBS_KEYS_TO_MODALITIES
from coro.common.utils.train_utils import get_device_from_parameters


class Algo(nn.Module):
    """
    Base algorithm class that all other algorithms subclass. Defines several
    functions that should be overriden by subclasses, in order to provide
    a standard API to be used by training functions such as @run_epoch in
    utils/train_utils.py.
    """
    name: str
    default_encoder_node: str | None

    def __init__(
        self,
        obs_config: ObsConfig,
        shape_meta: dict[str, tuple],
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            algo_config (Config object): instance of Config corresponding to the algo section
                of the config

            obs_config (Config object): instance of Config corresponding to the observation
                section of the config

            global_config (Config object): global training config

            obs_key_shapes (OrderedDict): dictionary that maps observation keys to shapes

            ac_dim (int): dimension of action space

            device (torch.Device): where the algo should live (i.e. cpu, gpu)
        """
        super().__init__()
        self.obs_config = obs_config
        self.shape_meta = shape_meta

        # normalization
        assert 'action' in shape_meta, "The canonical output key is 'action' and is required by all algorithms"
        self.input_shapes = {key: shape_meta[key] for key in all_keys(self.obs_config.modalities) if key in shape_meta}
        self.output_shapes = {'action': shape_meta['action']}
        self.dataset_stats = dataset_stats

        if self.dataset_stats is not None:
            input_normalization_modes = {}
            for key in self.input_shapes:
                input_normalization_modes[key] = get_normalization_mode(obs_config, key)

            self.output_shapes = {'action': shape_meta['action']}
            output_normalization_modes = {'action': obs_config.decoder.normalization_mode}

            self.normalize_inputs = Normalize(
                self.input_shapes, input_normalization_modes, dataset_stats
            )
            self.normalize_targets = Normalize(
                self.output_shapes, output_normalization_modes, dataset_stats
            )
            self.unnormalize_outputs = Unnormalize(
                self.output_shapes, output_normalization_modes, dataset_stats
            )
        else:
            self.normalize_inputs = lambda batch: batch
            self.normalize_targets = lambda batch: batch
            self.unnormalize_outputs = lambda batch: batch

        self.model = nn.Module()
        self._create_shapes()
        self._create_model()
        assert isinstance(self.model, nn.Module)

        self.reset()

    def _create_shapes(self):
        """
        Create obs_shapes, goal_shapes, and subgoal_shapes dictionaries, to make it
        easy for this algorithm object to keep track of observation key shapes. Each dictionary
        maps observation key to shape.

        Args:
            obs_keys (dict): dict of required observation keys for this training run (usually
                specified by the obs config), e.g., {"obs": ["rgb", "proprio"], "goal": ["proprio"]}
            obs_key_shapes (dict): dict of observation key shapes, e.g., {"rgb": [3, 224, 224]}
        """
        # determine shapes
        self.obs_shapes = OrderedDict()
        self.goal_shapes = OrderedDict()
        self.subgoal_shapes = OrderedDict()

        # We check across all modality groups (obs, goal, subgoal), and see if the inputted observation key exists
        # across all modalitie specified in the config. If so, we store its corresponding shape internally
        for k in self.input_shapes:
            if "obs" in self.obs_config.modalities and k in [obs_key for modality in self.obs_config.modalities.obs.values() for obs_key in modality]:
                self.obs_shapes[k] = self.shape_meta[k]
            if "goal" in self.obs_config.modalities and k in [obs_key for modality in self.obs_config.modalities.goal.values() for obs_key in
                                                              modality]:
                self.goal_shapes[k] = self.shape_meta[k]
            if "subgoal" in self.obs_config.modalities and k in [obs_key for modality in self.obs_config.modalities.subgoal.values() for obs_key in
                                                                 modality]:
                self.subgoal_shapes[k] = self.shape_meta[k]

    def _create_model(self):
        """
        Creates networks and places them into @self.nets.
        @self.nets should be a ModuleDict.
        """
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError

    def update(self):
        """
        Called at the end of each epoch.
        """
        pass

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        pass

    def get_delta_timestamps(self, fps: float) -> dict:
        return {}

    def prepare_batch(self, batch: dict) -> dict:
        """
        former pre_process:
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        former post_process
        Does some operations (like channel swap, uint8 to float conversion, normalization)
        after @process_batch_for_training is called, in order to ensure these operations
        take place on GPU.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader. Assumed to be on the device where
                training will occur (after @process_batch_for_training
                is called)

            obs_normalization_stats (dict or None): if provided, this should map observation
                keys to dicts with a "mean" and "std" of shape (1, ...) where ... is the
                default shape for the observation.

        Returns:
            batch (dict): postproceesed batch
        """
        def process_helper(obs_dict):
            obs_dict = process_obs_dict(obs_dict)
            if self.dataset_stats is not None:
                obs_dict = self.normalize_inputs(obs_dict)
            return obs_dict

        batch['obs'] = process_helper(batch['obs'])
        if 'goal_obs' in batch:
            batch['goal_obs'] = process_helper(batch['goal_obs'])
        if self.training:
            batch = self.normalize_targets(batch)
        return batch

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss log (dict): name -> summary statistic
        """
        log = OrderedDict()

        # record current optimizer learning rates
        for k in self.optimizers:
            for i, param_group in enumerate(self.optimizers[k].param_groups):
                log["Optimizer/{}{}_lr".format(k, i)] = param_group["lr"]

        return log

    def get_train_state(self):
        raise NotImplementedError

    def load_train_state(self):
        raise NotImplementedError


class PolicyAlgo(Algo):
    """
    Base class for all algorithms that can be used as policies.
    """
    def select_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        raise NotImplementedError


class ValueAlgo(Algo):
    """
    Base class for all algorithms that can learn a value function.
    """
    def get_state_value(self, obs_dict, goal_dict=None):
        """
        Get state value outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            value (torch.Tensor): value tensor
        """
        raise NotImplementedError

    def get_state_action_value(self, obs_dict, actions, goal_dict=None):
        """
        Get state-action value outputs.

        Args:
            obs_dict (dict): current observation
            actions (torch.Tensor): action
            goal_dict (dict): (optional) goal

        Returns:
            value (torch.Tensor): value tensor
        """
        raise NotImplementedError


class PlannerAlgo(Algo):
    """
    Base class for all algorithms that can be used for planning subgoals
    conditioned on current observations and potential goal observations.
    """
    def get_subgoal_predictions(self, obs_dict, goal_dict=None):
        """
        Get predicted subgoal outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoal prediction (dict): name -> Tensor [batch_size, ...]
        """
        raise NotImplementedError

    def sample_subgoals(self, obs_dict, goal_dict, num_samples=1):
        """
        For planners that rely on sampling subgoals.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoals (dict): name -> Tensor [batch_size, num_samples, ...]
        """
        raise NotImplementedError


class HierarchicalAlgo(Algo):
    """
    Base class for all hierarchical algorithms that consist of (1) subgoal planning
    and (2) subgoal-conditioned policy learning.
    """
    def select_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        raise NotImplementedError

    def get_subgoal_predictions(self, obs_dict, goal_dict=None):
        """
        Get subgoal predictions from high-level subgoal planner.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoal (dict): predicted subgoal
        """
        raise NotImplementedError

    @property
    def current_subgoal(self):
        """
        Get the current subgoal for conditioning the low-level policy

        Returns:
            current subgoal (dict): predicted subgoal
        """
        raise NotImplementedError


class LeRobotPolicy(PolicyAlgo):
    def __init__(
        self,
        obs_config: ObsConfig,
        shape_meta: dict[str, tuple],
        optimizer: Callable,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        lr_scheduler: Callable | None = None,
        grad_clip_norm: float | None = None,
        use_amp: bool = True,
    ):
        PolicyAlgo.__init__(self, obs_config, shape_meta, dataset_stats)

        if grad_clip_norm is None:
            grad_clip_norm = torch.inf
        self.grad_clip_norm = grad_clip_norm
        self.grad_scaler = GradScaler(enabled=use_amp)

        self.optimizer = self.lr_scheduler = None
        self._create_optimizer_and_scheduler(optimizer, lr_scheduler)

    def forward(self, batch, lock=None) -> dict:
        """Returns a dictionary of items for logging."""
        start_time = time.perf_counter()
        device = get_device_from_parameters(self)
        with torch.autocast(device_type=device.type) if self.use_amp else nullcontext():

            # LeRobot policies uses flat dictionaries as inputs, we 'flatten'
            # our structured batch and discard unused info, such as goal_obs
            flat_batch = batch['obs'] | {key: value for key, value in batch.items() if key != 'obs'}
            output_dict = self.compute_loss(flat_batch)
            # TODO(rcadene): policy.unnormalize_outputs(out_dict)
            loss = output_dict["loss"]
        self.grad_scaler.scale(loss).backward()

        # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
        self.grad_scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            self.grad_clip_norm,
            error_if_nonfinite=False,
        )

        # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        with lock if lock is not None else nullcontext():
            self.grad_scaler.step(self.optimizer)
        # Updates the scale for next iteration.
        self.grad_scaler.update()

        self.optimizer.zero_grad()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.update()

        info = {
            "loss": loss.item(),
            "grad_norm": float(grad_norm),
            "lr": self.optimizer.param_groups[0]["lr"],
            "update_s": time.perf_counter() - start_time,
            **{k: v for k, v in output_dict.items() if k != "loss"},
        }
        info.update({k: v for k, v in output_dict.items() if k not in info})

        return info

    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError

    def _create_optimizer_and_scheduler(self, optimizer: Callable, lr_scheduler: Callable | None):
        self.optimizer = optimizer(params=self.parameters())
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optimizer)

    def get_train_state(self):
        train_state = {"optimizer": self.optimizer.state_dict()}
        if self.lr_scheduler is not None:
            train_state = {"lr_scheduler": self.lr_scheduler.state_dict()}
        return train_state

    def load_train_state(self, train_state):
        self.optimizer.load_state_dict(train_state["optimizer"])
        if self.lr_scheduler is not None:
            self.optimizer.load_state_dict(train_state["lr_scheduler"])
        elif "lr_scheduler" in train_state:
            raise ValueError(
                "The checkpoint contains a lr_scheduler state_dict, but no LRScheduler was provided."
            )


