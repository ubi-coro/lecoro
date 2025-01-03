from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, Literal

from hydra_zen import store, builds, MISSING

from coro.common.utils.config_utils import to_snake_case, none


encoder_defaults = {
    'rgb/backbone': 'vit_small_patch16',
    'rgb/projection': 'spatial_softmax',
    'rgb/randomizer': ['crop_randomizer', 'color_randomizer'],
    'depth/randomizer': ['crop_randomizer']
}


def register_configs():
    """
    ================================================
    Build (partial) configs to instantiate encoders
    ================================================
    """
    from coro.common.config_gen.encoder import register_configs as register_encoders
    register_encoders()

    # build named, instantiable, partial configurations from the stored classes with their respective arguments
    from coro.common.utils.obs_utils import OBS_MODALITY_CLASSES
    from coro.common.config_gen.encoder import (
        OBS_BACKBONES,
        OBS_PROJECTIONS,
        OBS_NORMS,
        OBS_ACTIVATIONS,
        OBS_RANDOMIZERS,
    )
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
        for i in range(10):
            for name, config in randomizer_configs[m].items():
                store(config, group=f'observation/encoder/{m}/randomizer/{i+1}', name=to_snake_case(name))
            store(none, group=f'observation/encoder/{m}/randomizer/{i+1}', name='none')

            # store default randomizer for that modality and index, which is 'none' unless otherwise specified
            obs_defaults.append({f'observation/encoder/{m}/randomizer/{i+1}': 'none'})
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
    projection: Any = None
    norm: Any = None
    activation: Any = None
    randomizer: dict[str, Any] = field(default_factory=dict)
    normalization_stats: None | dict[str, list] = None
    normalization_mode: str = 'mean_std'

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



