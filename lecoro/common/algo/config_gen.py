import inspect
from dataclasses import dataclass, fields
from functools import partial
from typing import Callable

from hydra_zen import builds, store
from torch.optim import Adam

from lecoro.common.utils.config_utils import build_config_from_instance

# DO NOT MODIFY, will be filled automatically by calling @register_configs

# maps policy name (eg bc-small) onto its respective policy class (eg BC)
ALGO_NAME_TO_SPEC = {}

# maps the inverse, ie a policy class (BC) onto a list of registered policy names (bc-small, bc-large)
ALGO_CLS_TO_NAME = {}

# maps algo classes (eg BC) to a dictionary, which maps param types (eg Optimizer) onto
# the tuple (parameter_name_in_signature, default_value_for_that_algo)
ALGO_CLS_TO_NESTED_PARAMS = {}

# maps a param type (eg Optimizer) onto the set of configurations, such as [adam, sgd]
NESTED_PARAM_TO_CHOICES = {}

# maps a param type (eg Optimizer) onto the global default value for that param
NESTED_PARAM_TO_DEFAULTS = {}

ALL_PARAM_NAMES = set()


def register_variant(name=None, encoder_node=None, **kwargs):
    def _register_variant(algo_cls):
        global ALGO_NAME_TO_SPEC, ALGO_CLS_TO_NAME
        if name is None:
            _name = algo_cls.__name__
        else:
            _name = name

        if encoder_node is not None:
            kwargs['encoder_node'] = encoder_node

        # store mapping from algo name (ie bc-small) to policy class (ie BC)
        assert _name not in ALGO_NAME_TO_SPEC, f"Already registered policy {_name}!"
        ALGO_NAME_TO_SPEC[_name] = (algo_cls, kwargs)

        # store the reverse mapping
        if not algo_cls in ALGO_CLS_TO_NAME:
            ALGO_CLS_TO_NAME[algo_cls] = []
        ALGO_CLS_TO_NAME[algo_cls].append(_name)

        return algo_cls
    return _register_variant


def register_param(param_name, is_a=None, default_value=None):
    def _register_nested_param(algo):
        global NESTED_PARAM_TO_DEFAULTS, ALGO_CLS_TO_NESTED_PARAMS
        # inspect signature
        try:
            inferred_param_type, inferred_default = get_param_info(algo.__init__, param_name)
        except ValueError:
            #raise ValueError(f'{algo.__name__}: Unknown __init__ parameter {param_name}.')
            inferred_param_type, inferred_default = None, None

        # infer class and default value
        if is_a is not None:
            _param_type = is_a
        elif inferred_param_type is not None:
            _param_type = inferred_param_type
        else:
            raise ValueError(f'{algo.__name__}: Cannot infer type for param {param_name}, either type-hint or populate is_a.')

        # check that we know what this type is
        if _param_type not in NESTED_PARAM_TO_DEFAULTS:
            raise ValueError(f'{algo.__name__}: Unknown type {_param_type}, try running @register_configs first.')

        # grab default value
        if default_value is not None:
            _default_value = default_value
        elif inferred_default is not None:
            _default_value = inferred_default
        else:  # if nothing is specified, we grab the global default value for that parameter class
            _default_value = NESTED_PARAM_TO_DEFAULTS[_param_type]

        # store everything
        if algo not in ALGO_CLS_TO_NESTED_PARAMS:
            ALGO_CLS_TO_NESTED_PARAMS[algo] = {}
        if param_name in ALGO_CLS_TO_NESTED_PARAMS[algo]:
            raise ValueError(f'{algo}: Already registered param {param_name}.')
        ALGO_CLS_TO_NESTED_PARAMS[algo][param_name] = (_param_type, _default_value)
        ALL_PARAM_NAMES.add(param_name)
        return algo

    return _register_nested_param


def get_param_info(func, param_name):
    sig = inspect.signature(func)

    if param_name not in sig.parameters:
        raise ValueError(f'Unknown parameter {param_name}')
    param = sig.parameters[param_name]

    param_type = param.annotation if param.annotation != param.empty else None
    default_value = param.default if param.default != param.empty else None
    return param_type, default_value


def register_configs():
    # run all class definitions to register their configs
    import lecoro.common.algo.act.modeling_act
    import lecoro.common.algo.diffusion.modeling_diffusion
    import lecoro.common.algo.dit.modeling_dit

    # register all names, so what we can choose them easily via scripts/train.py +algo=bc-small
    stored_param_names_to_cls = {}
    algo_defaults = []
    for algo_name, algo_spec in ALGO_NAME_TO_SPEC.items():
        (algo_cls, algo_kwargs) = algo_spec

        # each algo defines a default encoder
        if 'encoder_node' in algo_kwargs:
            encoder_default = [dict(encoder=algo_kwargs['encoder_node'])]
            del algo_kwargs['encoder_node']
        elif algo_cls.default_encoder_node is not None:
            encoder_default = [dict(encoder=algo_cls.default_encoder_node)]
        else:
            encoder_default = []  # will use the global encoder default

        # register all group choices for each parameter of the algorithm
        algo_defaults = []
        _ALGO_CLS_TO_NESTED_PARAMS = ALGO_CLS_TO_NESTED_PARAMS.get(algo_cls, {})
        for param_name, (param_cls, default_value) in _ALGO_CLS_TO_NESTED_PARAMS.items():

            if param_name in stored_param_names_to_cls:  # we have seen this parameter name in another algorithm
                (_algo_cls, _param_cls) = stored_param_names_to_cls[param_name]
                if param_cls != _param_cls:
                    raise ValueError(
                        f"coro.common.config_gen.algo.register_configs: The parameter '{param_name}' is ambiguous! \n"
                        f"In {algo_cls}, this parameter refers to {param_cls}, but in {_algo_cls}, it refers to {_param_cls}. \n"
                        f"The names and types of registered parameters must be globally consistent for "
                        f"'algo/{param_name}: <example_value>' to work consistently!")
                #  continue
            else:
                stored_param_names_to_cls[param_name] = (algo_cls, param_cls)

            # Each named choice of this new parameter will be stored in its respective group
            for choice_name, param_cfg in NESTED_PARAM_TO_CHOICES[param_cls].items():
                store(param_cfg, group=f'algo/{param_name}', name=choice_name)

                if choice_name == '_default':
                    raise ValueError(
                        f'register_configs: _default is a reserved parameter name for the parameter {param_name}')

            # store default value for that parameter
            if param_name in algo_kwargs:  # default passed to register_variant, which overwrites the params default value
                default_value = algo_kwargs[param_name]
                del algo_kwargs[param_name]

            if isinstance(default_value, str):  # default value is a string referring to a group choice in config/algo_params.py
                _default_value = default_value
            elif type(default_value) == type:  # default_value is a class, that was passed to register_param
                default_cfg = builds(default_value, zen_partial=True, populate_full_signature=True)
                store(default_cfg, group=f'algo/{param_name}', name='_default')
                default_value = '_default'
            else:  # default_value is an instance, that was passed to register_param
                default_cfg = build_config_from_instance(default_value)
                store(default_cfg, group=f'algo/{param_name}', name='_default')
                default_value = '_default'
            algo_defaults.append({f'{param_name}': default_value})

        # now we need to build the algo config itself with its correct default values
        hydra_defaults = encoder_default + algo_defaults + ['_self_']
        algo_cfg = builds(algo_cls,
                          zen_partial=False,
                          populate_full_signature=True,
                          zen_meta={'name': algo_name},
                          zen_exclude=list(_ALGO_CLS_TO_NESTED_PARAMS.keys()),

                          # This one is important: when passing primitive types such as dicts or lists at construction+
                          # e.g., instantiate(algo_cfg, shape_meta=<some_dict>), we do NOT want shape_meta to be passed
                          # as a ListConfig, but as a primitive list instead. Otherwise, HuggingFace's config serialization
                          # ignores it.
                          hydra_convert='partial',
                          hydra_defaults=hydra_defaults,
                          **algo_kwargs)
        store(algo_cfg, group='algo', name=algo_name)
    return algo_defaults


@dataclass
class AlgoConfig:
    obs_config: dict[str, dict] = None
    dataset_stats: dict[str, dict] | None = None

    shape_meta: dict[str, tuple[int]] | None = None
    output_shapes: dict[str, tuple[int]] | None = None
    output_shapes: dict[str, tuple[int]] | None = None


@dataclass
class LeRobotConfig(AlgoConfig):
    optimizer: Callable = partial(Adam, lr=0.0003, weight_decay=1e-4)
    lr_scheduler: Callable | None = None
    grad_clip_norm: float | None = None
    use_amp: bool = True

    def items(self):
        """Returns a view object that displays a list of dataclass fields and their values."""
        return ((field.name, getattr(self, field.name)) for field in fields(self))


if __name__ == "__main__":
    from typing import Type

    def example_function(a: Type[int], b: str = "example_default"):
        pass

    type_annotation, default_value = get_param_info(example_function, 'a')
    print(type_annotation, default_value)
