import importlib
import os
from dataclasses import dataclass
from typing import Any

from hydra_zen import builds, MISSING, store
from omegaconf import OmegaConf

from lecoro.common.algo.encoder.config_gen import ObsConfig
from lecoro.common.utils.config_utils import (
    build_nested_dataclass_config,
    resolve_debug_suffix,
    resolve_overrides_and_ts,
    resolve_ts
)
from lecoro.common.datasets.utils import remove_modality_prefix


CONFIGS_REGISTERED = False


@dataclass
class EvalConfig:
    enable: bool = False
    frequency: int = 1000
    batch_size: int = 50
    n_episodes: int = 50
    use_async_envs: bool = False


@dataclass
class CheckpointConfig:
    enable: bool = True
    frequency: int = 2000


@dataclass
class LogConfig:
    frequency: int = 100
    enable_system_metrics_logging: bool = True

@dataclass
class WandBConfig:
    enable: bool = True
    disable_artifact: bool = False  # Set to true to disable saving an artifact despite save_checkpoint == True
    project: str = 'ubi'
    notes: str = ''

@dataclass
class TrainConfig:
    offline_steps: int = 10_000
    batch_size: int = 256
    num_workers: int = 4
    device: str = 'cuda'

    dataset: Any = MISSING

    # place this inside policy
    drop_n_last_frames: int = 7  # ${algo.horizon} - ${algo.n_action_steps} - ${algo.n_obs_steps} + 1

@dataclass
class Config:
    algo: Any = MISSING
    workspace: Any = MISSING
    observation: ObsConfig = MISSING
    env: Any = MISSING
    training: TrainConfig = MISSING

    eval: EvalConfig = MISSING
    checkpoint: CheckpointConfig = MISSING
    logging: LogConfig = MISSING
    wandb: WandBConfig = MISSING

    resume: bool = MISSING
    seed: int = MISSING
    root: str | None = None
    dataset_repo_id: str | list[str] = MISSING
    video_backend: str = MISSING
    debug: bool = False


def register_configs():
    global CONFIGS_REGISTERED
    if CONFIGS_REGISTERED:
        return

    # run register functions to store each sub-config in the store
    from lecoro.common.algo.config_gen import register_configs as register_algos
    from lecoro.common.algo.algo_params import register_configs as register_algo_params
    from lecoro.common.algo.encoder.config_gen import register_configs as register_encoder

    # register all resolvers
    OmegaConf.register_new_resolver("timestamp", resolve_ts)  # use ${ts}
    OmegaConf.register_new_resolver('debug-suffix', resolve_debug_suffix)  # use ${debug-suffix:${debug}}
    OmegaConf.register_new_resolver('overrides-and-ts', resolve_overrides_and_ts)  # use ${overrides-and-ts:}

    # start registering configs
    store._overwrite_ok = True

    # store the one dataset key formatter we have
    formatter_cfg = builds(remove_modality_prefix, return_modality=False, zen_partial=True)
    store(formatter_cfg, group='training/dataset/key_formatter', name='remove_prefix')

    register_algo_params()
    register_algos()
    empty_obs_defaults = register_encoder()

    config = build_nested_dataclass_config(Config, hydra_defaults=empty_obs_defaults)
    store(config, name='_template_')

    store.add_to_hydra_store(overwrite_ok=True)
    CONFIGS_REGISTERED = True


def import_all_except(folder_path, exclude_files):
    """
    Imports all .py files in a folder except those specified in exclude_files.

    Parameters:
    - folder_path (str): Path to the folder containing .py files.
    - exclude_files (list): List of filenames (without .py extension) to exclude from importing.

    Returns:
    - modules (dict): Dictionary of imported modules with filename as the key.
    """
    modules = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.py'):
            module_name = file_name[:-3]  # Remove the .py extension
            if module_name not in exclude_files:
                module_path = os.path.join(folder_path, file_name)
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                modules[module_name] = module  # Add to dictionary of modules
    return modules


if __name__ == "__main__":
    register_configs()

