from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn
from hydra_zen import store, builds

from lecoro.common.utils.config_utils import to_snake_case, none

# DO NOT MODIFY, will be filled automatically by calling @register_configs

# Nested mappings from modalities to all available group choices [modality -> name -> (cls, kwargs)]
OBS_BACKBONES = {}
OBS_PROJECTIONS = {}
OBS_NORMS = {}  # hardcoded
OBS_ACTIVATIONS = {}  # hardcoded
OBS_RANDOMIZERS = {}

DEFAULT_BACKBONES = {}
DEFAULT_SHARE_BACKBONES = {}
DEFAULT_DROPOUT = {}
DEFAULT_POOLINGS = {}
DEFAULT_NORMS = {}
DEFAULT_ACTIVATIONS = {}
DEFAULT_RANDOMIZERS = {}
MAX_NUM_RANDOMIZERS = 10

encoder_defaults = {
    'rgb/backbone': 'resnet18',
    'rgb/projection': 'spatial_softmax',
    'depth/backbone': 'resnet18',
    'depth/projection': 'spatial_softmax',
    'tactile/backbone': 'resnet18',
    'tactile/projection': 'spatial_softmax',
}


def register_configs():
    """
    ================================================
    Build (partial) configs to instantiate encoders
    ================================================
    """
    global OBS_BACKBONES, OBS_PROJECTIONS, OBS_RANDOMIZERS, OBS_ACTIVATIONS, OBS_RANDOMIZERS

    # Build empty dictionary for each modality
    from lecoro.common.utils.obs_utils import OBS_MODALITY_CLASSES  # this runs all modality definitions and registers their names
    all_modalities = list(OBS_MODALITY_CLASSES.keys())

    for m in all_modalities:
        if m not in OBS_BACKBONES:
            OBS_BACKBONES[m] = {}
        if m not in OBS_PROJECTIONS:
            OBS_PROJECTIONS[m] = {}
        if m not in OBS_RANDOMIZERS:
            OBS_RANDOMIZERS[m] = {}

    # Import all backbones, projections and randomizer to register them, we do this here to avoid circular imports
    import lecoro.common.algo.encoder.backbone
    import lecoro.common.algo.encoder.pooling
    import lecoro.common.algo.encoder.randomizer

    # register modality-agnostic normalization layers and activation functions manually
    global OBS_NORMS, OBS_ACTIVATIONS
    hydra_partial = dict(populate_full_signature=True, zen_partial=True)
    hydra_full = dict(populate_full_signature=True)

    OBS_ACTIVATIONS = {
        'relu': builds(nn.ReLU, **hydra_full),
        'rrelu': builds(nn.RReLU, **hydra_full),
        'hardtanh': builds(nn.Hardtanh, **hydra_full),
        'tanh': builds(nn.Tanh, **hydra_full),
        'silu': builds(nn.SiLU, **hydra_full),
        'mish': builds(nn.Mish, **hydra_full),
        'hardswish': builds(nn.Hardswish, **hydra_full),
        'elu': builds(nn.ELU, **hydra_full),
        'celu': builds(nn.CELU, **hydra_full),
        'selu': builds(nn.SELU, **hydra_full),
        'glu': builds(nn.GLU, **hydra_full),
        'gelu': builds(nn.GELU, **hydra_full),
        'identity': builds(nn.Identity, **hydra_full),
        'hardshrink': builds(nn.Hardshrink, **hydra_full),
        'leakyrelu': builds(nn.LeakyReLU, **hydra_full),
        'logsigmoid': builds(nn.LogSigmoid, **hydra_full),
        'softplus': builds(nn.Softplus, **hydra_full),
        'softshrink': builds(nn.Softshrink, **hydra_full),
        'prelu': builds(nn.PReLU, **hydra_full),
        'softsign': builds(nn.Softsign, **hydra_full),
        'tanhshrink': builds(nn.Tanhshrink, **hydra_full),
        'softmin': builds(nn.Softmin, **hydra_full),
        'softmax': builds(nn.Softmax, **hydra_full)
    }

    OBS_NORMS = {
        'local_response': builds(nn.LocalResponseNorm, **hydra_partial),
        'crossmap_lrn2d': builds(nn.CrossMapLRN2d, **hydra_partial),
        'layer': builds(nn.LayerNorm, **hydra_partial),
        'rms': builds(nn.RMSNorm, **hydra_partial),
    }

    # Build named, instantiable, partial configurations from the stored classes with their respective arguments
    from lecoro.common.utils.obs_utils import OBS_MODALITY_CLASSES
    all_modalities = list(OBS_MODALITY_CLASSES.keys())

    backbone_configs = {}
    projection_configs = {}
    norm_configs = {}
    activation_configs = {}
    randomizer_configs = {}
    for m in all_modalities:
        backbone_configs[m] = {}
        for name, (cls, kwargs) in OBS_BACKBONES[m].items():
            backbone_configs[m][name] = builds(cls, zen_partial=True, populate_full_signature=True, **kwargs)

        projection_configs[m] = {}
        for name, (cls, kwargs) in OBS_PROJECTIONS[m].items():
            projection_configs[m][name] = builds(cls, zen_partial=True, populate_full_signature=True, **kwargs)

        norm_configs[m] = deepcopy(OBS_NORMS)

        activation_configs[m] = deepcopy(OBS_ACTIVATIONS)

        randomizer_configs[m] = {}
        for name, (cls, kwargs) in OBS_RANDOMIZERS[m].items():
            randomizer_configs[m][name] = builds(cls, zen_partial=True, populate_full_signature=True, **kwargs)

    """
    ================================================
    Store all encoder configs and build defaults
    ================================================
    """
    obs_defaults = []
    for m in all_modalities:
        # store all backbone configs
        for name, config in backbone_configs[m].items():
            store(config, group=f'observation/encoder/{m}/backbone', name=to_snake_case(name))
        store(none, group=f'observation/encoder/{m}/backbone', name='none')

        # store default encoder for that modality, which is 'none' unless otherwise specified
        obs_defaults.append({f'observation/encoder/{m}/backbone': 'none'})

        # store all projection configs
        for name, config in projection_configs[m].items():
            store(config, group=f'observation/encoder/{m}/projection', name=to_snake_case(name))
        store(none, group=f'observation/encoder/{m}/projection', name='none')

        # store default encoder for that modality, which is 'none' unless otherwise specified
        obs_defaults.append({f'observation/encoder/{m}/projection': 'none'})

        # store all norm configs
        for name, config in norm_configs[m].items():
            store(config, group=f'observation/encoder/{m}/norm', name=to_snake_case(name))
        store(none, group=f'observation/encoder/{m}/norm', name='none')

        # store default encoder for that modality, which is 'none' unless otherwise specified
        obs_defaults.append({f'observation/encoder/{m}/norm': 'none'})

        # store all activation configs
        for name, config in activation_configs[m].items():
            store(config, group=f'observation/encoder/{m}/activation', name=to_snake_case(name))

        # store default encoder for that modality, which is 'none' unless otherwise specified
        obs_defaults.append({f'observation/encoder/{m}/activation': 'identity'})

        # store all available named randomizer configs
        #   we want users to define lists of randomizers, which are executed in sequence
        #   OmegaConf does not support direct list composition atm, so we resort to
        #   an ordered dictionary with a fixed number of possible randomizers
        for i in range(MAX_NUM_RANDOMIZERS):
            for name, config in randomizer_configs[m].items():
                store(config, group=f'observation/encoder/{m}/randomizer/{i + 1}', name=to_snake_case(name))
            store(none, group=f'observation/encoder/{m}/randomizer/{i + 1}', name='none')

            # store default randomizer for that modality and index, which is 'none' unless otherwise specified
            obs_defaults.append({f'observation/encoder/{m}/randomizer/{i + 1}': 'none'})
    return obs_defaults


"""
================================================
Dataclass configs
================================================
"""


@dataclass
class ObsKeys:
    low_dim: list = field(default_factory=list)
    rgb: list = field(default_factory=list)
    depth: list = field(default_factory=list)
    scan: list = field(default_factory=list)
    tactile: list = field(default_factory=list)
    audio: list = field(default_factory=list)


@dataclass
class ModalityConfig:  # modality keys, uses lists in case of multiples, i.e, multi-cam setups
    # environments can emit specific keys and policies can digest specific keys (or everything in one modality independent of the keys)
    obs: ObsKeys = ObsKeys
    goal: ObsKeys = ObsKeys
    subgoal: ObsKeys = ObsKeys


@dataclass
class EncoderCoreConfig:
    backbone: Any = None
    dropout: float | None = None
    projection: Any = None
    norm: Any = None
    activation: Any = None
    randomizer: dict[str, Any] = field(default_factory=dict)
    normalization_stats: dict[str, list] | None = None
    normalization_mode: str = 'mean_std'
    share_backbone: bool = False

    def __post_init__(self):
        if self.normalization_mode not in ('mean_std', 'min_max'):
            raise ValueError("EncoderCoreConfig: normalization_mode must be in ('mean_std', 'min_max')")


@dataclass
class EncoderConfig:
    low_dim: EncoderCoreConfig = EncoderCoreConfig
    rgb: EncoderCoreConfig = EncoderCoreConfig
    depth: EncoderCoreConfig = EncoderCoreConfig
    scan: EncoderCoreConfig = EncoderCoreConfig
    tactile: EncoderCoreConfig = EncoderCoreConfig
    audio: EncoderCoreConfig = EncoderCoreConfig


@dataclass
class DecoderConfig:  # overwrite modality config for a specific key
    normalization_stats: None | dict[str, list] = None
    normalization_mode: str = 'mean_std'


@dataclass
class ObsConfig:
    modalities: ModalityConfig = ModalityConfig
    decoder: DecoderConfig = DecoderConfig
    encoder: EncoderConfig = EncoderConfig
    encoder_overwrites: dict[str, EncoderCoreConfig] = field(default_factory=dict)  #: Optional[dict] = None


"""
================================================
Register Functions
================================================
"""


def register_backbone(modality: str | list[str] | None = None, name: str | None = None, **kwargs):  # used as a decorator
    def _register_backbone(target_backbone):
        if name is None:
            _name = target_backbone.__name__
        else:
            _name = name

        if modality is None:
            _modality = list(OBS_BACKBONES.keys())
        else:
            _modality = modality

        if not isinstance(_modality, list):
            _modality = [_modality]

        for m in _modality:
            if m not in OBS_BACKBONES:
                OBS_BACKBONES[m] = {}
            assert _name not in OBS_BACKBONES[m], f"Already registered obs backbone {_name} for modality {m}!"
            OBS_BACKBONES[m][_name] = (target_backbone, kwargs)
        return target_backbone

    return _register_backbone


def register_projection(modality: str | list[str] | None = None, name: str | None = None, **kwargs):  # used as a decorator
    def _register_projection(target_projection):
        if name is None:
            _name = target_projection.__name__
        else:
            _name = name

        if modality is None:
            _modality = list(OBS_PROJECTIONS.keys())
        else:
            _modality = modality

        if not isinstance(_modality, list):
            _modality = [_modality]

        for m in _modality:
            if m not in OBS_PROJECTIONS:
                OBS_PROJECTIONS[m] = {}
            assert _name not in OBS_PROJECTIONS[m], f"Already registered obs projection {_name} for modality {m}!"
            OBS_PROJECTIONS[m][_name] = (target_projection, kwargs)
        return target_projection

    return _register_projection


def register_randomizer(modality: str | list[str] | None = None, name: str | None = None, **kwargs):  # used as a decorator
    def _register_randomizer(target_randomizer):
        if name is None:
            _name = target_randomizer.__name__
        else:
            _name = name

        if not modality or modality is None:
            _modality = list(OBS_RANDOMIZERS.keys())
        else:
            _modality = modality

        if not isinstance(_modality, list):
            _modality = [_modality]

        for m in _modality:
            if m not in OBS_RANDOMIZERS:
                OBS_RANDOMIZERS[m] = {}
            assert _name not in OBS_RANDOMIZERS[m], f"Already registered obs randomizer {_name} for modality {m}!"
            OBS_RANDOMIZERS[m][_name] = (target_randomizer, kwargs)
        return target_randomizer

    return _register_randomizer


if __name__ == "__main__":
    register_configs()
