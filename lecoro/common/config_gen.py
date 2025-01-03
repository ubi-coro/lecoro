import os
import importlib
from dataclasses import dataclass
from typing import Any

from hydra_zen import MISSING, store

from coro.common.config_gen.experiment import ExperimentConfig
from coro.common.config_gen.observation import ObsConfig
from coro.common.config_gen.training import TrainConfig
from coro.common.config_gen.eval import EvalConfig
from coro.common.utils.config_utils import build_nested_dataclass_config

# # fix dynamic modulem loading, then this should work

# todo: clean up algo selection


def resolve_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def resolve_debug_suffix(is_debug: bool) -> str:
    if is_debug:
        return '-debug'
    else:
        return ''


def resolve_overrides_and_ts() -> str:
    """
    def _resolver(cfg: Config):
        return f'{get_hydra_overrides()}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
    return _resolver()
    """
    return f'[{get_hydra_overrides()}]_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

@dataclass
class EvalConfig:
    enable: bool = False
    frequency: int = 1000
    batch_size: int = 50
    n_episodes: int = 50


@dataclass
class CheckpointConfig:
    enable: bool = True
    frequency: int = 2000
    batch_size: int = 50
    n_episodes: int = 50


@dataclass
class LogConfig:
    frequency: int = 100
    enable_system_metrics_logging: bool = True


@dataclass
class TrainConfig:
    resume: bool = '${resume}'
    seed: bool = '${seed}'

    # data
    dataset: Any = MISSING

    offline_steps: int = 10_000
    num_workers: int = 4
    device: str = 'cuda'

    batch_size: int = 256

    # place this inside policy
    drop_n_last_frames: int = 7  # ${algo.horizon} - ${algo.n_action_steps} - ${algo.n_obs_steps} + 1

    eval: EvalConfig = EvalConfig
    checkpoint: CheckpointConfig = CheckpointConfig
    logging: LogConfig = LogConfig

@dataclass
class Config:
    algo: Any = MISSING
    recorder: Any = MISSING
    observation: ObsConfig = MISSING
    env: Any = MISSING
    training: TrainConfig = MISSING
    hydra: Any = MISSING
    wandb: Any = MISSING

    resume: bool = MISSING
    seed: int = MISSING
    root: str | None = None
    dataset_repo_id: str | list[str] = MISSING
    video_backend: str = MISSING
    debug: bool = False


def register_configs():

    # run register functions to store each sub-config in the store
    from coro.common.config_gen.algo import register_configs as register_algos
    from coro.common.config_gen.algo_params import register_configs as register_algo_params
    from coro.common.config_gen.observation import register_configs as register_obs
    from coro.common.config_gen.resolver import register_resolver
    from coro.common.config_gen.training import register_configs as register_training

    # register all resolvers
    OmegaConf.register_new_resolver("timestamp", resolve_ts)  # use ${ts}
    OmegaConf.register_new_resolver('debug-suffix', resolve_debug_suffix)  # use ${debug-suffix:${debug}}
    OmegaConf.register_new_resolver('overrides-and-ts', resolve_overrides_and_ts)  # use ${overrides-and-ts:}

    # start registering configs
    store._overwrite_ok = True

    # dataset key formatter
    from coro.common.utils.dataset_utils import remove_modality_prefix
    formatter_cfg = builds(remove_modality_prefix, return_modality=False, zen_partial=True)
    store(formatter_cfg, group='training/dataset/key_formatter', name='remove_prefix')

    register_algo_params()
    register_algos()
    empty_obs_defaults = register_obs()

    # store internal hydra defaults
    hydra_default = {
        "run": {
            "dir": "${oc.env:HOME}/outputs${debug-suffix:${debug}}/${now:%Y-%m-%d}/${now:%H-%M-%S}_${hydra.job.name}"
        },
        "sweep": {
            "dir": "${oc.env:HOME}/outputs${debug-suffix:${debug}}/${now:%Y-%m-%d}/${now:%H-%M-%S}_${hydra.job.name}",
            "subdir": "${hydra.job.num}"
        },
        "job": {
            "config": {
                "override_dirname": {
                    "exclude_keys": ['experiment.name']
                }
            },
            "name": "${experiment.name}"
        }
    }
    config = build_nested_dataclass_config(Config, hydra_defaults=empty_obs_defaults)
    store(config, name='_base_')

    store.add_to_hydra_store(overwrite_ok=True)


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

