import logging
import warnings
from collections import OrderedDict
from functools import partial
import textwrap
from typing import Callable, Literal, Optional

import einops
import torch
import torch.nn as nn

import coro.common.utils.obs_utils as ObsUtils
from coro.common.model.encoder.randomizer import Randomizer
from coro.common.model.encoder.pooling import Pooling, Flatten


def compose_encoder_overwrites(key, overwrites):
    return {}


class ObservationEncoder(nn.Module):
    """for any given observation outputs 2d features of the form [@n_tokens, @feature_dim]"""

    def __init__(self, feature_dim: int = None, feature_aggregation: Literal['concat', 'mean', 'sequence'] = 'flatten'):
        assert feature_dim is not None or feature_aggregation == 'concat', \
            f"ObservationEncoder: 'feature_dim' can only be None if feature_aggregation='concat', but got '{feature_aggregation}'"
        super().__init__()
        self._locked = False

        if feature_aggregation == 'concat':
            self._aggregation = _concat
        elif feature_aggregation == 'mean':
            self._aggregation = _mean
        elif feature_aggregation == 'sequence':
            self._aggregation = _sequence
        else:
            raise ValueError(f"ObservationEncoder: Unknown 'feature_aggregation' {feature_aggregation}")

        # these dicts map keys ('cam_left', 'proprio') to respective net, randomizers etc
        self.in_shapes = OrderedDict()
        self.out_shapes = OrderedDict()
        self.backbones = nn.ModuleDict()
        self.final_layers = nn.ModuleDict()
        self.randomizers = nn.ModuleDict()
        self.share_backbone = OrderedDict()
        self.is_image = OrderedDict()  # store image keys for faster lookup

        self.feature_dim = feature_dim
        self.feature_aggregation = feature_aggregation
        self._order = 2 if feature_aggregation == 'sequence' else 1  # order of the output tensor

    def register_obs_key(
            self,
            name,
            shape,
            backbone=None,
            backbone_kwargs=None,

            pooling=None,
            pooling_kwargs=None,

            norm=None,
            activation=None,
            randomizers=None,
            share_backbone_from=None,
    ):
        is_image_key = ObsUtils.is_image_key(name)
        assert not self._locked, "ObservationEncoder: @register_obs_key called after @make"
        assert name not in self.in_shapes, f"ObservationEncoder: modality {name} already exists"
        assert not is_image_key or len(shape) == 3, f"ObservationEncoder: all image modalities must have shape (C, H, W)"

        if pooling_kwargs is None:
            pooling_kwargs = dict()

        backbone, randomizers, out_shape = self._build_backbone(shape, backbone, backbone_kwargs, randomizers, share_backbone_from)

        # add pooling layers to match self.feature_dim, add normalization and non-linearity
        layers = []
        if len(out_shape) == 3:  # backbone outputs a feature map [D, H, W]
            if self._order == 2:  # [D, H, W] -> [@feature_dim, H, W], will use einops during inference to get [H * W, @feature_dim]
                layers.append(nn.Conv2d(out_shape[0], self.feature_dim, kernel_size=1))
                out_shape = [self.feature_dim] + out_shape[1:3]

            elif pooling is not None:  # [D, H, W] -> [@feature_dim,] / [D * H * W,]
                if isinstance(pooling, partial) or isinstance(pooling, Callable):
                    pooling = pooling(input_shape=out_shape, feature_dim=self.feature_dim, **pooling_kwargs)

                assert isinstance(pooling, Pooling), 'All image poolings must subclass coro.common.model.encoder.pooling.Pooling!'
                layers.append(pooling)
                out_shape = pooling.output_shape(input_shape=out_shape)

            else:  # [D, H, W] -> [@feature_dim, ] / [D * H * W,]
                pooling = Flatten(input_shape=out_shape, feature_dim=self.feature_dim)
                layers.append(pooling)
                out_shape = pooling.output_shape(input_shape=out_shape)

        elif len(out_shape) == 2:  # backbone outputs a sequence (e.g., ViT) [S, D]
            if self._order == 2:  # [S, D] -> [@feature_dim, D]
                layers.append([nn.Linear(out_shape[1], self.feature_dim)])
                out_shape = [out_shape[0], self.feature_dim]
            else:  # [S, D] -> [@feature_dim, ] / [S * D,]
                pooling = Flatten(input_shape=out_shape, feature_dim=self.feature_dim)
                layers.append(pooling)
                out_shape = pooling.output_shape(input_shape=out_shape)

        elif self.feature_dim is not None:  # backbone outputs a feature vector [D,] -> [@feature_dim,]
            layers.append(nn.Linear(out_shape[0], self.feature_dim))
            out_shape = [self.feature_dim]

        # norm
        # todo: check if we can have channels for group norm
        if norm is not None:
            if isinstance(norm, str):
                ...  # load from string
            norm = norm(out_shape[0])
            layers.append(norm)

        if activation is None:
            activation = nn.Identity
        if isinstance(norm, str):
            ...  # load from string
        layers.append(activation)

        self.in_shapes[name] = shape
        self.out_shapes[name] = out_shape
        self.backbones[name] = backbone
        self.final_layers[name] = nn.Sequential(*layers)
        self.randomizers[name] = nn.ModuleList(randomizers)
        self.share_backbone[name] = share_backbone_from
        self.is_image[name] = is_image_key

    def make(self):
        """
        Creates the encoder networks and locks the encoder so that more modalities cannot be added.
        """
        assert not self._locked, "ObservationEncoder: @make called more than once"
        self._create_layers()
        self._locked = True

    def _create_layers(self):
        """
        Creates all networks and layers required by this encoder using the registered modalities.
        """
        assert not self._locked, "ObservationEncoder: layers have already been created"

        for key in self.in_shapes:
            if self.share_backbone[key] is not None:
                # make sure net is shared with another modality
                self.backbones[key] = self.backbone[self.share_backbone[key]]

    def forward(self, obs_dict):
        """
        Processes modalities according to the ordering in @self.in_shapes. For each
        modality, it is processed with a randomizer (if present), an encoder
        network (if present), and again with the randomizer (if present), flattened,
        and then concatenated with the other processed modalities.

        Args:
            obs_dict (OrderedDict): dictionary that maps modalities to torch.Tensor
                batches that agree with @self.in_shapes. All modalities in
                @self.in_shapes must be present, but additional modalities
                can also be present.

        Returns:
            feats (torch.Tensor): flat features of shape [B, D]
        """
        assert self._locked, "ObservationEncoder: @make has not been called yet"

        # ensure all modalities that the encoder handles are present
        assert set(self.in_shapes.keys()).issubset(obs_dict), \
            f"ObservationEncoder: {list(obs_dict.keys())} does not contain all modalities {list(self.in_shapes.keys())}"

        # process keys in order given by @self.in_shapes
        features = []  # (@self.n_tokens, b, self.feature_dim)
        for key in self.in_shapes:

            # process obs with pooling layer
            x = self.backbone(key=key, x=obs_dict[key])

            x = self.final_layer(key=key, x=x)

            if len(x.shape) == 4:
                x = einops.rearrange(x, "b c h w -> (h w) b c")
            if len(x.shape) == 3:
                features.extend(x)
            else:
                features.append(x)

        return self._aggregation(features)

    def output_shape(self, input_shape=None):
        """
        Compute the output shape of the encoder.
        """
        if self.feature_aggregation == 'sequence':
            n_tokens = sum([shape[1] * shape[2] for name, shape in self.out_shapes.items() if len(shape) == 3])
            n_tokens += sum([shape[0] for name, shape in self.out_shapes.items() if len(shape) == 2])
            n_tokens += sum([1 for name, shape in self.out_shapes.items() if len(shape) == 1])
            return [n_tokens, self.feature_dim]
        elif self.feature_aggregation == 'mean':
            return [self.feature_dim]
        elif self.feature_aggregation == 'concat':
            return [sum([shape[0] for name, shape in self.out_shapes.items()])]

    def backbone(self, key, x):
        # maybe process encoder input with randomizer
        for rand in self.randomizers[key]:
            if rand is not None:
                x = rand.forward_in(x)

        # maybe process with backbone
        if self.backbones[key] is not None:
            x = self.backbones[key](x)

        return x

    def final_layer(self, key, x):
        x = self.final_layers[key](x)

        # maybe process encoder output with randomizer
        for rand in self.randomizers[key]:
            if rand is not None:
                x = rand.forward_out(x)
        return x

    def _build_backbone(
            self,
            shape,
            backbone=None,
            backbone_kwargs=None,
            randomizers=None,
            share_backbone_from=None,
    ):
        # share processing with another modality
        if share_backbone_from is not None:
            assert share_backbone_from in self.in_shapes, f"ObservationEncoder: unknown obs_key {share_backbone_from}"

        if backbone_kwargs is None:
            backbone_kwargs = dict()

        # handle randomizer
        out_shape = shape
        if randomizers is not None:
            randomizers = randomizers if isinstance(randomizers, list) else [randomizers]  # might not be list
            for i, rand in enumerate(randomizers):
                if isinstance(rand, partial):
                    rand = rand(input_shape=shape)
                assert isinstance(rand, Randomizer), f"ObservationEncoder: all randomizers must subclass @Randomizer"
                out_shape = rand.output_shape_in(shape)
                randomizers[i] = rand

        # instantiate net if needed
        if isinstance(backbone, partial) or isinstance(backbone, Callable):
            backbone = backbone(input_shape=out_shape, **backbone_kwargs)

        if backbone is not None:
            assert isinstance(backbone, nn.Module)
            out_shape = backbone.output_shape(input_shape=out_shape)

        return backbone, randomizers, out_shape



    def __repr__(self):
        """
        Pretty print the encoder.
        """
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        for k in self.in_shapes:
            msg += textwrap.indent('\nKey(\n', ' ' * 4)
            indent = ' ' * 8
            msg += textwrap.indent("name={}\nshape={}\n".format(k, self.in_shapes[k]), indent)
            msg += textwrap.indent("modality={}\n".format(ObsUtils.OBS_KEYS_TO_MODALITIES[k]), indent)
            msg += textwrap.indent("randomizer={}\n".format(self.randomizers[k]), indent)
            msg += textwrap.indent("net={}\n".format(self.backbones[k]), indent)
            msg += textwrap.indent("sharing_from={}\n".format(self.obs_share_mods[k]), indent)
            msg += textwrap.indent(")", ' ' * 4)
        msg += textwrap.indent("\noutput_shape={}".format(self.output_shape()), ' ' * 4)
        msg = header + '(' + msg + '\n)'
        return msg


def encoder_factory(
        obs_shapes: dict[str, tuple[int]],
        feature_dim: int | None = None,
        feature_aggregation: Literal['concat', 'mean', 'sequence'] = 'concat',
        encoder_cls: ObservationEncoder = ObservationEncoder,
        encoder_overwrites: Optional['OverwriteConfig'] = None
):
    """
    Utility function to create an @ObservationEncoder from kwargs specified in config.
    """
    if encoder_overwrites is None:
        encoder_overwrites = dict()

    enc = encoder_cls(feature_dim=feature_dim, feature_aggregation=feature_aggregation)
    assert isinstance(enc, ObservationEncoder)

    for k, obs_shape in obs_shapes.items():
        obs_modality = ObsUtils.OBS_KEYS_TO_MODALITIES[k]

        kwargs = dict(
            name=k,
            shape=obs_shape,
            backbone=ObsUtils.DEFAULT_BACKBONES[obs_modality],
            pooling=ObsUtils.DEFAULT_POOLINGS[obs_modality],
            activation=ObsUtils.DEFAULT_ACTIVATIONS[obs_modality],
            randomizers=ObsUtils.DEFAULT_RANDOMIZERS[obs_modality]
        )

        if k in encoder_overwrites:
            kwargs.update(compose_encoder_overwrites(k, encoder_overwrites))

        enc.register_obs_key(**kwargs)

    enc.make()
    return enc


def _concat(features: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(features, dim=-1)


def _sequence(features: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(features, dim=0)


def _mean(features: list[torch.Tensor]) -> torch.Tensor:
    return _sequence(features).mean(0)
