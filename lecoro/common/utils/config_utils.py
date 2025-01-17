import datetime
import functools
import re
from pathlib import Path

import hydra
from hydra_zen import make_config, builds, ZenField, MISSING


# can be used to store non as a default parameter when manually adding to hydra.ConfigStore
# hydra needs a callable to store configs
def return_none():
    return None


none = builds(return_none)


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


def to_snake_case(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


def build_nested_dataclass_config(config, hydra_defaults):
    zen_config = []
    for value in config.__dataclass_fields__.values():
        item = (
            ZenField(name=value.name, hint=value.type, default=value.default)
            if value.default is not MISSING
            else ZenField(name=value.name, hint=value.type)
        )
        zen_config.append(item)

    return make_config(*zen_config, bases=(config,), hydra_defaults=["_self_"] + hydra_defaults)


def build_config_from_instance(instance):
    if instance is None:
        return none

    if isinstance(instance, functools.partial):
        return builds(
            instance.func,  # The function being partially applied
            *instance.args,  # Positional arguments
            **instance.keywords,  # Keyword arguments
            zen_partial=True,
            populate_full_signature=True
        )
    else:
        instance_dict = instance.__dict__
        return builds(
            instance.__class__,
            **instance_dict,
            zen_partial=True,
            populate_full_signature=True
        )


def get_hydra_cwd() -> Path:
    return Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)


def get_hydra_overrides() -> str:
    return hydra.core.hydra_config.HydraConfig.get().job.override_dirname

