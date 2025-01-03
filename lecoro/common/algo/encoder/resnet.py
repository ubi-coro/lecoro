# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math

import numpy as np
import torch
import torch.nn as nn
from r3m import load_r3m
from torchvision import models

from coro.common.model.base_nets import BaseEncoder, CoordConv2d
from coro.common.utils.obs_utils import register_encoder

"""
================================================
ResNet Factories
================================================
"""


@register_encoder(modality='rgb')
def resnet34(input_shape, norm_cfg={"name": "batch_norm"}, **kwargs):
    model = ResNet(input_shape, size=34, norm_cfg=norm_cfg, **kwargs)
    return model


@register_encoder(modality='rgb')
def resnet50(input_shape, norm_cfg={"name": "batch_norm"}, **kwargs):
    model = ResNet(input_shape, size=50, norm_cfg=norm_cfg, **kwargs)
    return model


@register_encoder(modality='rgb')
def r3m_resnet18(input_shape, **kwargs):
    model = R3M(input_shape, size=18, **kwargs)
    return model


@register_encoder(modality='rgb')
def r3m_resnet34(input_shape, **kwargs):
    model = R3M(input_shape, size=34, **kwargs)
    return model


@register_encoder(modality='rgb')
def r3m_resnet50(input_shape, **kwargs):
    model = R3M(input_shape, size=50, **kwargs)
    return model


@register_encoder(modality='rgb')
def robomimic_resnet18(input_shape, norm_cfg={"name": "batch_norm"}, **kwargs):
    model = RobomimicResNet(input_shape, size=18, norm_cfg=norm_cfg, **kwargs)
    return model


@register_encoder(modality='rgb')
def robomimic_resnet34(input_shape, norm_cfg={"name": "batch_norm"}, **kwargs):
    model = RobomimicResNet(input_shape, size=34, norm_cfg=norm_cfg, **kwargs)
    return model


@register_encoder(modality='rgb')
def robomimic_resnet50(input_shape, norm_cfg={"name": "batch_norm"}, **kwargs):
    model = RobomimicResNet(input_shape, size=50, norm_cfg=norm_cfg, **kwargs)
    return model


"""
================================================
ResNet Wrappers
================================================
"""


@register_encoder(
    modality='rgb',
    name='resnet18',
    size=18,
    norm_cfg={"name": "batch_norm"}
)
class ResNet(BaseEncoder):
    def __init__(
            self,
            input_shape,
            size,
            norm_cfg,
            weights=None,
            restore_path="",
            avg_pool=True,
            conv_repeat=0,
    ):
        norm_layer = _make_norm(norm_cfg)
        model = _construct_resnet(size, norm_layer, weights)
        model.fc = nn.Identity()
        if not avg_pool:
            model.avgpool = nn.Identity()

        if conv_repeat > 1:
            w = model.conv1.weight.data.repeat((1, conv_repeat, 1, 1))
            model.conv1.weight.data = w
            model.conv1.in_channels *= conv_repeat

        super().__init__(input_shape, model, restore_path)
        self._size, self._avg_pool = size, avg_pool

    def forward(self, x):
        if self._avg_pool:
            return self._model(x)[:, None]
        B = x.shape[0]
        x = self._model(x)
        x = x.reshape((B, self.embed_dim, -1))
        return x.transpose(1, 2)

    @property
    def embed_dim(self):
        return {18: 512, 34: 512, 50: 2048}[self._size]

    @property
    def n_tokens(self):
        if self._avg_pool:
            return 1
        return 49  # assuming 224x224 images

    def output_shape(self, input_shape):
        return [1, self.embed_dim]


class R3M(ResNet):
    def __init__(self, input_shape, size, avg_pool=True):
        model = load_r3m(f"resnet{size}").module.convnet.cpu()
        if not avg_pool:
            model.avgpool = nn.Identity()
        super(BaseEncoder).__init__(input_shape, model, False)
        self._size, self._avg_pool = size, avg_pool


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.
    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """

    def __init__(
            self,
            input_shape,
            num_kp=None,
            temperature=1.0,
            learnable_temperature=False,
            output_variance=False,
            noise_std=0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not use spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution,
            and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape  # (C, H, W)

        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter("temperature", temperature)
        else:
            # temperature held constant after initialization
            temperature = nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer("temperature", temperature)

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert len(input_shape) == 3
        assert input_shape[0] == self._in_c
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial
        probability distribution is created using a softmax, where the support is the
        pixel locations. This distribution is used to compute the expected value of
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.
        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert feature.shape[1] == self._in_c
        assert feature.shape[2] == self._in_h
        assert feature.shape[3] == self._in_w
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = torch.nn.functional.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1]
        # for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution
            # and compute second-order statistics to return
            expected_xx = torch.sum(
                self.pos_x * self.pos_x * attention, dim=1, keepdim=True
            )
            expected_yy = torch.sum(
                self.pos_y * self.pos_y * attention, dim=1, keepdim=True
            )
            expected_xy = torch.sum(
                self.pos_x * self.pos_y * attention, dim=1, keepdim=True
            )
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2]
            # where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(
                -1, self._num_kp, 2, 2
            )
            feature_keypoints = (feature_keypoints, feature_covar)

        return feature_keypoints


class RobomimicResNet(BaseEncoder):
    def __init__(
            self,
            input_shape,
            size,
            norm_cfg,
            restore_path="",
            weights=None,
            img_size=224,
            feature_dim=64,
            input_coord_conv=False
    ):
        # Build resnet
        norm_layer = _make_norm(norm_cfg)
        model = _construct_resnet(size, norm_layer, weights)

        # Maybe modify first layer
        input_channel = input_shape[0]
        if input_coord_conv:
            model.conv1 = CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif input_channel != 3:
            model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Cut the last two layers.
        resnet = nn.Sequential(*(list(model.children())[:-2]))
        resnet_out_dim = int(math.ceil(img_size / 32.0))
        resnet_output_shape = [512, resnet_out_dim, resnet_out_dim]

        # Add 2d softmax and pooling
        spatial_softmax = SpatialSoftmax(
            resnet_output_shape,
            num_kp=64,
            temperature=1.0,
            noise_std=0.0,
            output_variance=False,
            learnable_temperature=False,
        )
        pool_output_shape = spatial_softmax.output_shape(resnet_output_shape)

        flatten = nn.Flatten(start_dim=1, end_dim=-1)
        proj = nn.Linear(int(np.prod(pool_output_shape)), feature_dim)
        model = nn.Sequential(resnet, spatial_softmax, flatten, proj)

        # Maybe restore this model from path
        super().__init__(input_shape, model, restore_path)
        self.feature_dim = feature_dim
        self.input_coord_conv = input_coord_conv

    def forward(self, x):
        return self._model(x)[:, None]

    @property
    def embed_dim(self):
        return self.feature_dim

    @property
    def n_tokens(self):
        return 1

    def output_shape(self, input_shape):
        return [self.embed_dim]


def _make_norm(norm_cfg):
    if norm_cfg["name"] == "batch_norm":
        return nn.BatchNorm2d
    if norm_cfg["name"] == "group_norm":
        num_groups = norm_cfg["num_groups"]
        return lambda num_channels: nn.GroupNorm(num_groups, num_channels)
    if norm_cfg["name"] == "diffusion_policy":
        def _gn_builder(num_channels):
            num_groups = int(num_channels // 16)
            return nn.GroupNorm(num_groups, num_channels)

        return _gn_builder
    raise NotImplementedError(f"Missing norm layer: {norm_cfg['name']}")


def _construct_resnet(size, norm, weights=None):
    if size == 18:
        w = models.ResNet18_Weights
        m = models.resnet18(norm_layer=norm)
    elif size == 34:
        w = models.ResNet34_Weights
        m = models.resnet34(norm_layer=norm)
    elif size == 50:
        w = models.ResNet50_Weights
        m = models.resnet50(norm_layer=norm)
    else:
        raise NotImplementedError(f"Missing size: {size}")

    if weights is not None:
        w = w.verify(weights).get_state_dict(progress=True)
        if norm is not nn.BatchNorm2d:
            w = {
                k: v
                for k, v in w.items()
                if "running_mean" not in k and "running_var" not in k
            }
        m.load_state_dict(w)
    return m


if __name__ == '__main__':
    # Define a random input tensor (batch_size=1, channels=3, height=224, width=224)
    input_tensor = torch.randn(1, 3, 224, 224)

    # ResNet Tests
    print("Testing ResNet models...")

    resnet34_model = resnet34()
    resnet34_output = resnet34_model(input_tensor)
    print(f"ResNet34 output shape: {resnet34_output.shape}")

    resnet50_model = resnet50()
    resnet50_output = resnet50_model(input_tensor)
    print(f"ResNet50 output shape: {resnet50_output.shape}")

    # R3M Tests
    print("Testing R3M models...")
    r3m_resnet18_model = r3m_resnet18()
    r3m_resnet18_output = r3m_resnet18_model(input_tensor)
    print(f"R3M ResNet18 output shape: {r3m_resnet18_output.shape}")

    r3m_resnet34_model = r3m_resnet34()
    r3m_resnet34_output = r3m_resnet34_model(input_tensor)
    print(f"R3M ResNet34 output shape: {r3m_resnet34_output.shape}")

    r3m_resnet50_model = r3m_resnet50()
    r3m_resnet50_output = r3m_resnet50_model(input_tensor)
    print(f"R3M ResNet50 output shape: {r3m_resnet50_output.shape}")

    # RoboMimicResNet Tests
    print("Testing RoboMimic ResNet models...")
    robomimic_resnet18_model = robomimic_resnet18()
    robomimic_resnet18_output = robomimic_resnet18_model(input_tensor)
    print(f"RoboMimic ResNet18 output shape: {robomimic_resnet18_output.shape}")

    robomimic_resnet34_model = robomimic_resnet34()
    robomimic_resnet34_output = robomimic_resnet34_model(input_tensor)
    print(f"RoboMimic ResNet34 output shape: {robomimic_resnet34_output.shape}")

    robomimic_resnet50_model = robomimic_resnet50()
    robomimic_resnet50_output = robomimic_resnet50_model(input_tensor)
    print(f"RoboMimic ResNet50 output shape: {robomimic_resnet50_output.shape}")
