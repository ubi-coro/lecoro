from dataclasses import dataclass
from hydra_zen import builds
from typing import Any, Tuple, Optional

import diffusers
import torch.optim as optim
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.schedulers import SchedulerMixin
from torch.optim.lr_scheduler import LambdaLR

from lecoro.common.utils.config_utils import none


def register_configs():
    from lecoro.common.algo.config_gen import NESTED_PARAM_TO_CHOICES, NESTED_PARAM_TO_DEFAULTS
    hydra_full = dict(populate_full_signature=True)
    hydra_partial = dict(zen_partial=True, **hydra_full)
    """
    ================================================
    Store all optimizers
    ================================================
    """
    NESTED_PARAM_TO_CHOICES[optim.Optimizer] = {
        'adam': builds(optim.Adam, lr=0.001, weight_decay=1e-4, **hydra_partial),
        'adamw': builds(optim.AdamW, lr=0.01, betas=[0.95, 0.999], eps=1e-8, weight_decay=1e-4, **hydra_partial),
        'sgd': builds(optim.SGD, lr=0.01, momentum=0.9, weight_decay=1e-4, **hydra_partial)
    }
    NESTED_PARAM_TO_DEFAULTS[optim.Optimizer] = 'adam'

    """
    ================================================
    Store all learning rate schedulers
    ================================================
    """
    NESTED_PARAM_TO_CHOICES[optim.lr_scheduler.LRScheduler] = {
        'cosine-annealing': builds(get_scheduler, name='cosine', num_training_steps='${...training.offline_steps}', num_warmup_steps=0, **hydra_partial),
        'cosine-with-warmup': builds(get_scheduler, name='cosine', num_training_steps='${...training.offline_steps}', num_warmup_steps=500, **hydra_partial),
        'none': none
    }
    NESTED_PARAM_TO_DEFAULTS[optim.lr_scheduler.LRScheduler] = 'none'

    """
    ================================================
    Store all vae priors
    ================================================
    """
    NESTED_PARAM_TO_CHOICES[PriorConfig] = {
        'gaussian': builds(PriorConfig, **hydra_full),
        'gmm': builds(PriorConfig, use_gmm=True, **hydra_full),
        'categorical': builds(PriorConfig, use_categorical=True, **hydra_full),
        'learned-gaussian': builds(PriorConfig, learn=True, is_conditioned=True, **hydra_full),
        'learned-gmm': builds(PriorConfig, learn=True, is_conditioned=True, use_gmm=True, gmm_learn_weights=True, **hydra_full),
        'learned-categorical': builds(PriorConfig, learn=True, is_conditioned=True, use_categorical=True, **hydra_full),
    }
    NESTED_PARAM_TO_DEFAULTS[PriorConfig] = 'gaussian'

    """
    ================================================
    Store all noise schedulers for diffusion
    ================================================
    """
    NESTED_PARAM_TO_CHOICES[SchedulerMixin] = {
        'ddpm': builds(
            DDPMScheduler,
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
            **hydra_full
        ),
        'ddim': builds(
            DDIMScheduler,
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon',
            **hydra_full
        ),
    }
    NESTED_PARAM_TO_DEFAULTS[SchedulerMixin] = 'ddpm'

    """
    ================================================
    Store all noise schedulers for diffusion
    ================================================
    """
    NESTED_PARAM_TO_CHOICES[EMAModel] = {
        'default': builds(EMAModel, power=0.75, **hydra_partial),
        'none': none
    }
    NESTED_PARAM_TO_DEFAULTS[EMAModel] = 'default'

    """
    ================================================
    Store different unets for diffusion
    ================================================
    """
    from coro.common.model.diffusion_nets import ConditionalUnet1D  # import here to avoid circular import
    NESTED_PARAM_TO_CHOICES[ConditionalUnet1D] = {
        'default': builds(ConditionalUnet1D, **hydra_partial),
    }
    NESTED_PARAM_TO_DEFAULTS[ConditionalUnet1D] = 'default'

def get_scheduler(
    name: str,
    optimizer: optim.Optimizer,
    step_rules: Optional[str] = None,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[Any] = None,  # we need to loosen this type hint to enable interpolation
    num_cycles: int = 1,
    power: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    return diffusers.optimization.get_scheduler(
        name,
        optimizer,
        step_rules,
        num_warmup_steps,
        num_training_steps,
        num_cycles,
        power,
        last_epoch
    )

@dataclass
class PriorConfig:
    learn: bool = False  # learn Gaussian / GMM prior instead of N(0, 1)
    layer_dims: Tuple[int] = (300, 400)  # prior MLP layer dimensions (if learning conditioned prior)
    is_conditioned: bool = False  # whether to condition prior on observations
    use_gmm: bool = False  # whether to use GMM prior
    gmm_num_modes: int = 10  # number of GMM modes
    gmm_learn_weights: bool = False  # whether to learn GMM weights
    use_categorical: bool = False  # whether to use categorical prior
    categorical_dim: int = 10  # the number of categorical classes for each latent dimension
    categorical_gumbel_softmax_hard: bool = False  # use hard selection in forward pass
    categorical_init_temp: float = 1.0  # initial gumbel-softmax temp
    categorical_temp_anneal_step: float = 0.001  # linear temp annealing rate
    categorical_min_temp: float = 0.3  # lowest gumbel-softmax temp

