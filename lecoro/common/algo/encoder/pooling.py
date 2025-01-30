import abc

import numpy as np
import torch
import torch.nn as nn

from lecoro.common.algo.encoder.config_gen import register_projection
from lecoro.common.utils.obs_utils import get_image_modalities


class Pooling(nn.Module):
    def __init__(self, input_shape, feature_dim):
        super().__init__()
        assert len(input_shape) == 3
        self._input_shape = input_shape
        self._feature_dim = feature_dim

    @abc.abstractmethod
    def output_shape(self, input_shape=None):
        raise NotImplementedError

    @property
    def feature_dim(self):
        return self._feature_dim


@register_projection(modality=get_image_modalities())
class SpatialSoftmax(Pooling):
    """
    Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """
    def __init__(
        self,
        input_shape,
        feature_dim,
        num_kp=32,
        temperature=1.,
        learnable_temperature=False,
        output_variance=False,
        noise_std=0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not using spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        self._in_c, self._in_h, self._in_w = input_shape  # (C, H, W)

        if num_kp is not None:
            input_projection = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            input_projection = None
            self._num_kp = self._in_c

        if feature_dim is None:
            feature_dim = 2 * self._num_kp
        super().__init__(input_shape, feature_dim)

        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std
        self.input_projection = input_projection
        self.output_projection = nn.Linear(2 * self._num_kp, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self._in_w),
                np.linspace(-1., 1., self._in_h)
                )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={}, temperature={}, noise={})'.format(
            self._num_kp, self.temperature.item(), self.noise_std)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(input_shape[0] == self._in_c)
        return [self._feature_dim]

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
        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.input_projection is not None:
            feature = self.input_projection(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = self.softmax(feature / self.temperature)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        # flatten keypoint coordinates
        feature_keypoints = torch.flatten(feature_keypoints, start_dim=1)

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()

        return self.output_projection(torch.flatten(feature_keypoints, start_dim=1))


@register_projection()
class SpatialMeanPool(Pooling):
    """
    Module that averages inputs across all spatial dimensions (dimension 2 and after),
    leaving only the batch and channel dimensions.
    """
    def __init__(self, input_shape, feature_dim):
        super().__init__(input_shape, feature_dim)
        self.projection = nn.Linear(np.prod(input_shape[:1]), feature_dim)

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return [self.in_shape[:1]]  # [C, H, W] -> [C]

    def forward(self, x):
        """Forward pass - average across all dimensions except batch and channel."""
        x = torch.flatten(x, start_dim=2).mean(dim=2)
        return self.projection(x)


@register_projection()
class Flatten(Pooling):
    def __init__(self, input_shape: tuple[int], feature_dim: int | None = None):
        if feature_dim is None:
            projection = nn.Identity()
            feature_dim = np.prod(input_shape)
        else:
            projection = nn.Linear(np.prod(input_shape), feature_dim)
        super().__init__(input_shape, feature_dim)
        self.projection = projection

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return [self.feature_dim]  # [C, H, W] -> [C]

    def forward(self, x):
        """Forward pass - flatten all dimensions except batch."""
        x = torch.flatten(x, start_dim=1)
        return self.projection(x)


