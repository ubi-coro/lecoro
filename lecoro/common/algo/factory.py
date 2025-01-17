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
import inspect
import logging

from hydra_zen import get_target, instantiate
from omegaconf import DictConfig, OmegaConf

from lecoro.common.algo.algo_protocol import Algo
from lecoro.common.config_gen import Config, ObsConfig


def _policy_cfg_from_hydra_cfg(policy_cfg_class, hydra_cfg):
    expected_kwargs = set(inspect.signature(policy_cfg_class).parameters)
    if not set(hydra_cfg.policy).issuperset(expected_kwargs):
        logging.warning(
            f"Hydra config is missing arguments: {set(expected_kwargs).difference(hydra_cfg.policy)}"
        )

    # OmegaConf.to_container returns lists where sequences are found, but our dataclasses use tuples to avoid
    # issues with mutable defaults. This filter changes all lists to tuples.
    def list_to_tuple(item):
        return tuple(item) if isinstance(item, list) else item

    policy_cfg = policy_cfg_class(
        **{
            k: list_to_tuple(v)
            for k, v in OmegaConf.to_container(hydra_cfg.policy, resolve=True).items()
            if k in expected_kwargs
        }
    )
    return policy_cfg


def make_algo(
    cfg: Config,
    shape_meta: dict[str, tuple[int]] | None = None,
    pretrained_algo_name_or_path: str | None = None,
    dataset_stats=None
) -> Algo:
    """Make an instance of a algo class.

    Args:
        cfg: A parsed Hydra configuration (see scripts). If `pretrained_algo_name_or_path` is
            provided, only `hydra_cfg.algo._target_` is used while everything else is ignored.
        shape_meta: A dictionary mapping observation keys to their expected shapes.
        pretrained_algo_name_or_path: Either the repo ID of a model hosted on the Hub or a path to a
            directory containing weights saved using `Policy.save_pretrained`. Note that providing this
            argument overrides everything in `hydra_cfg.algo` apart from `hydra_cfg.algo.name`.
        dataset_stats: Dataset statistics to use for (un)normalization of inputs/outputs in the algo. Must
            be provided when initializing a new algo, and must not be provided when loading a pretrained
            algo. Therefore, this argument is mutually exclusive with `pretrained_algo_name_or_path`.
    """
    if not (pretrained_algo_name_or_path is None) ^ (shape_meta is None or dataset_stats is None):
        raise ValueError(
            "Exactly one of `pretrained_algo_name_or_path` and [`shape_meta`, `dataset_stats`]  must be provided."
        )

    if pretrained_algo_name_or_path is None:
        # Make a fresh algo
        algo = instantiate(cfg.algo, obs_config=cfg.observation, shape_meta=shape_meta, dataset_stats=dataset_stats)
    else:
        # Load a pretrained algo and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        # TODO(alexander-soare): This hack makes use of huggingface_hub's tooling to load the algo with,
        # pretrained weights which are then loaded into a fresh algo with the desired config. This PR in
        # huggingface_hub should make it possible to avoid the hack:
        # https://github.com/huggingface/huggingface_hub/pull/2274.
        algo_cls = get_target(cfg.algo)
        algo_pretrained: Algo = algo_cls.from_pretrained(pretrained_algo_name_or_path, obs_config=cfg.observation)
        algo = instantiate(cfg.algo, shape_meta=algo_pretrained.config.shape_meta, obs_config=cfg.observation)
        algo.load_state_dict(algo_pretrained.state_dict())

    return algo
