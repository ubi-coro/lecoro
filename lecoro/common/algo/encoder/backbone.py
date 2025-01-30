import abc
import math
from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
from timm.models.vision_transformer import resize_pos_embed
from torchvision import models as vision_models
from torchvision import transforms
from torchvision.ops.misc import FrozenBatchNorm2d

from lecoro.common.algo.encoder.config_gen import register_backbone
from lecoro.common.algo.base_nets import CoordConv2d, ShallowConv


"""
================================================
Register Backbones
================================================
"""

@register_backbone(modality='rgb')
def resnet18(input_shape, norm_cfg={"name": "batch_norm"}, **kwargs):
    model = ResNet(input_shape, size=18, norm_cfg=norm_cfg, **kwargs)
    return model


@register_backbone(modality='rgb')
@register_backbone(name='resnet34_pretrained', modality='rgb', weights="ResNet34_Weights.IMAGENET1K_V1")
def resnet34(input_shape, norm_cfg={"name": "batch_norm"}, **kwargs):
    model = ResNet(input_shape, size=34, norm_cfg=norm_cfg, **kwargs)
    return model


@register_backbone(modality='rgb')
@register_backbone(name='resnet50_pretrained', modality='rgb', weights="ResNet50_Weights.IMAGENET1K_V1")
def resnet50(input_shape, norm_cfg={"name": "batch_norm"}, **kwargs):
    model = ResNet(input_shape, size=50, norm_cfg=norm_cfg, **kwargs)
    return model


@register_backbone(modality='rgb')
def r3m_resnet18(input_shape, **kwargs):
    model = R3M(input_shape, size=18, **kwargs)
    return model


@register_backbone(modality='rgb')
def r3m_resnet34(input_shape, **kwargs):
    model = R3M(input_shape, size=34, **kwargs)
    return model


@register_backbone(modality='rgb')
def r3m_resnet50(input_shape, **kwargs):
    model = R3M(input_shape, size=50, **kwargs)
    return model


@register_backbone(modality='rgb', name='vit_small_patch16', patch_size=16, embed_dim=384, depth=12, num_heads=6)
@register_backbone(modality='rgb', name='vit_base_patch16', patch_size=16, embed_dim=768, depth=12, num_heads=12)
@register_backbone(modality='rgb', name='vit_large_patch16', patch_size=16, embed_dim=1024, depth=24, num_heads=16)
@register_backbone(modality='rgb', name='vit_huge_patch14', patch_size=14, embed_dim=1280, depth=32, num_heads=16)
def vit(input_shape, patch_size, embed_dim, depth, num_heads, **kwargs):
    model = VisionTransformer(
        input_shape,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


"""
================================================
Backbone Classes
================================================
"""

class Backbone(nn.Module):
    def __init__(self, input_shape, model, restore_path):
        super().__init__()
        self._model = model
        self._input_shape = input_shape
        if restore_path:
            print("Restoring model from", restore_path)
            state_dict = torch.load(restore_path, map_location="cpu")
            state_dict = (
                state_dict["features"]
                if "features" in state_dict
                else state_dict["model"]
            )
            self.load_state_dict(state_dict)

    @abc.abstractmethod
    def output_shape(self, input_shape=None):
        raise NotImplementedError

    def forward(self, x):
        out = self._model(x)
        if list(self.output_shape(list(x.shape)[1:])) != list(out.shape)[1:]:
            raise ValueError(
                f'Size mismatch: expect size {str(self.output_shape(list(x.shape)[1:]))}, but got size {str(list(out.shape)[1:])}')
        return out


class ShallowConv(Backbone):
    """
    A shallow convolutional encoder used in TDMPC
    """
    def __init__(self, input_shape=3, num_channel=32, restore_path=""):
        self.num_channel = num_channel
        model = nn.Sequential(
            torch.nn.Conv2d(input_shape[0], num_channel, kernel_size=7, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(num_channel, num_channel, kernel_size=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        super(ShallowConv, self).__init__(input_shape, model, restore_path)

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
        # this is propably wrong
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._input_channel)
        out_h = int(math.floor(input_shape[1] / 2.))
        out_w = int(math.floor(input_shape[2] / 2.))
        return [self._output_channel, out_h, out_w]


@register_backbone(name='resnet18_pretrained', modality='rgb', weights="ResNet18_Weights.IMAGENET1K_V1")
class ResNet(Backbone):
    """
    A ResNet18 block that can be used to process input images.
    """
    def __init__(
        self,
        input_shape,
        size=18,
        norm_cfg={"name": "batch_norm"},
        weights=None,
        frozen=False,
        restore_path="",
        replace_final_stride_with_dilation=False,
        conv_repeat=0,
        input_coord_conv=False,
    ):
        """
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            pretrained (bool): if True, load pretrained weights for all ResNet layers.
            input_coord_conv (bool): if True, use a coordinate convolution for the first layer
                (a convolution where input channels are modified to encode spatial pixel location)
        """
        norm_layer = _make_norm(norm_cfg)
        model = _construct_resnet(size, norm_layer, weights, replace_final_stride_with_dilation)

        assert not (input_coord_conv and frozen), 'You cannot a new layer for a frozen pretrained model'

        if input_coord_conv:
            model.conv1 = CoordConv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif input_shape[0] != 3:  # channel
            model.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)

        if conv_repeat > 1:
            w = model.conv1.weight.data.repeat((1, conv_repeat, 1, 1))
            model.conv1.weight.data = w
            model.conv1.in_channels *= conv_repeat

        if frozen:
            for param in model.parameters():
                param.requires_grad = False

        # cut the last fc layer
        self._size = size
        self._input_coord_conv = input_coord_conv
        model = torch.nn.Sequential(*(list(model.children())[:-2]))
        super().__init__(input_shape, model, restore_path)

    @property
    def embed_dim(self):
        return {18: 512, 34: 512, 50: 2048}[self._size]

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
        assert(len(input_shape) == 3)
        out_h = int(math.ceil(input_shape[1] / 32.))
        out_w = int(math.ceil(input_shape[2] / 32.))
        return [self.embed_dim, out_h, out_w]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={}, input_coord_conv={})'.format(self._input_channel, self._input_coord_conv)


class R3M(Backbone):
    """
    Base class for ConvNets pretrained with R3M (https://arxiv.org/abs/2203.12601)
    """
    def __init__(
        self,
        input_shape,
        size='resnet18',
        frozen=True,
        preprocess=False
    ):
        """
        Using R3M pretrained observation encoder network proposed by https://arxiv.org/abs/2203.12601
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            r3m_model_class (str): select one of the r3m pretrained model "resnet18", "resnet34" or "resnet50"
            freeze (bool): if True, use a frozen R3M pretrained model.
        """
        try:
            from r3m import load_r3m
        except ImportError:
            print("WARNING: could not load r3m library! Please follow https://github.com/facebookresearch/r3m to install R3M")

        model = load_r3m(f"resnet{size}").module.convnet.cpu()

        assert input_shape[0] == 3  # R3M only support input image with channel size 3
        assert size in [18, 34, 50]  # make sure the selected r3m model do exist

        if preprocess:
            preprocess = nn.Sequential(
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )
            model = nn.Sequential(*([preprocess] + list(model.module.convnet.children())))

        if frozen:
            for param in model.parameters():
                param.requires_grad = False

        super().__init__(input_shape, model, False)

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
        assert(len(input_shape) == 3)
        return [self.embed_dim, 1, 1]


class MVP(Backbone):
    """
    Base class for ConvNets pretrained with MVP (https://arxiv.org/abs/2203.06173)
    """
    def __init__(
        self,
        input_shape,
        mvp_model_class='vitb-mae-egosoup',
        frozen=True,
    ):
        """
        Using MVP pretrained observation encoder network proposed by https://arxiv.org/abs/2203.06173
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            mvp_model_class (str): select one of the mvp pretrained model "vits-mae-hoi", "vits-mae-in", "vits-sup-in", "vitb-mae-egosoup" or "vitl-256-mae-egosoup"
            freeze (bool): if True, use a frozen MVP pretrained model.
        """
        try:
            import mvp
        except ImportError:
            print("WARNING: could not load mvp library! Please follow https://github.com/ir413/mvp to install MVP.")

        model = mvp.load(mvp_model_class)

        assert input_shape[0] == 3  # MVP only support input image with channel size 3
        assert mvp_model_class in ["vits-mae-hoi", "vits-mae-in", "vits-sup-in", "vitb-mae-egosoup", "vitl-256-mae-egosoup"] # make sure the selected r3m model do exist

        if '256' in mvp_model_class:
            input_img_size = 256
        else:
            input_img_size = 224
        self.preprocess = nn.Sequential(
            transforms.Resize(input_img_size)
        )

        if frozen:
            for param in model.parameters():
                param.requires_grad = False
        super().__init__(input_shape, model, False)

    def forward(self, x):
        out = self._model(self.preprocess(x))
        if list(self.output_shape(list(x.shape)[1:])) != list(out.shape)[1:]:
            raise ValueError(
                f'Size mismatch: expect size {str(self.output_shape(list(x.shape)[1:]))}, but got size {str(list(out.shape)[1:])}')
        return out

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
        assert(len(input_shape) == 3)
        if 'vitb' in self._mvp_model_class:
            output_shape = [768]
        elif 'vitl' in self._mvp_model_class:
            output_shape = [1024]
        else:
            output_shape = [384]
        return output_shape

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={}, input_coord_conv={}, pretrained={}, freeze={})'.format(self._input_channel, self._input_coord_conv, self._pretrained, self._freeze)


class VisionTransformer(Backbone):
    """VisionTransformer that outputs the 1D cls token embedding"""
    def __init__(
            self,
            input_shape,
            # custom vit params
            global_pool=False,
            use_cls=True,
            mask_ratio=None,
            del_head=True,
            restore_path="",
            # timm vit params
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_path_rate=0.0,
            **kwargs
    ):
        assert use_cls and not global_pool, "token counting only works for use_cls mode"
        model = timm.models.vision_transformer.VisionTransformer(img_size=input_shape[1:3], patch_size=patch_size,
                                                    **kwargs)
        super().__init__(input_shape, model, restore_path=False)  # we handle restoring from path later
        self._model.classifier_feature = "use_cls_token" if use_cls else "global_pool"
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        if del_head:
            del self._model.head  # Remove classification head

        if self._model.classifier_feature == "global_pool":
            norm_layer = norm_layer
            embed_dim = embed_dim
            self._model.fc_norm = norm_layer(embed_dim)

            del self._model.norm  # remove the original norm

        if self._model.classifier_feature == "reshape_embedding":
            self._model.final_spatial = int(self.patch_embed.num_patches ** 0.5)
            self._model.embed_dim = (
                self.patch_embed.grid_size[0],
                self.patch_embed.grid_size[1],
                embed_dim,
            )

        # Restore from path if provided
        self._model = _load_vit(self._model, restore_path)
        self._embed_dim = embed_dim

    def handle_outcome(self, x):
        if self._model.classifier_feature == "global_pool":
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self._model.fc_norm(x)
        elif self._model.classifier_feature == "use_cls_token":
            x = self._model.norm(x)
            outcome = x[:, :1]  # use cls token
        elif self._model.classifier_feature == "reshape_embedding":
            x = self._model.norm(x)
            outcome = _reshape_embedding(
                x[:, 1:]
            )  # remove cls token and reshape embedding
        else:
            raise NotImplementedError("Unknown classifier feature type.")

        return outcome

    def forward_features(self, x):
        B = x.shape[0]
        x = self._model.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self._model.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if self.mask_ratio is not None:
            x, _, _ = _random_masking(x, mask_ratio=self.mask_ratio)

        # append cls token
        cls_token = self._model.cls_token + self._model.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(B, -1, -1), x), dim=1)

        x = self._model.blocks(x)
        return self.handle_outcome(x)[:, 0, :]  # remove sequence dimension because we only care about the cls token anyway

    def forward(self, x):
        return self.forward_features(x)  # tokens are dim 0

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def n_tokens(self):
        # hard-coded assuming use_cls_token
        return 1

    def output_shape(self, input_shape):
        if len(input_shape) == 4:
            B, C, H, W = input_shape
        else:
            C, H, W = input_shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        num_patches = patch_h * patch_w

        if self._model.classifier_feature == "use_cls_token":
            return [self._model.embed_dim]
        elif self._model.classifier_feature == "global_pool":
            return [num_patches, self._model.embed_dim]
        else:
            raise NotImplementedError("Unknown classifier feature type.")


def _construct_resnet(size, norm, weights=None, replace_final_stride_with_dilation=False):
    replace_stride_with_dilation = [False, False, replace_final_stride_with_dilation]
    if size == 18:
        w = vision_models.ResNet18_Weights
        m = vision_models.resnet18(weights=w, norm_layer=norm, replace_stride_with_dilation=replace_stride_with_dilation)
    elif size == 34:
        w = vision_models.ResNet34_Weights
        m = vision_models.resnet34(weights=w, norm_layer=norm, replace_stride_with_dilation=replace_stride_with_dilation)
    elif size == 50:
        w = vision_models.ResNet50_Weights
        m = vision_models.resnet50(weights=w, norm_layer=norm, replace_stride_with_dilation=replace_stride_with_dilation)
    else:
        raise NotImplementedError(f"Missing size: {size}")
    return m

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


def _make_norm(norm_cfg):
    if norm_cfg["name"] == "batch_norm":
        return nn.BatchNorm2d
    if norm_cfg["name"] == "frozen_batch_norm":
        return FrozenBatchNorm2d
    if norm_cfg["name"] == "group_norm":
        num_groups = norm_cfg["num_groups"]
        return lambda num_channels: nn.GroupNorm(num_groups, num_channels)
    if norm_cfg["name"] == "diffusion_policy":
        def _gn_builder(num_channels):
            num_groups = int(num_channels // 16)
            return nn.GroupNorm(num_groups, num_channels)
        return _gn_builder

    raise NotImplementedError(f"Missing norm layer: {norm_cfg['name']}")


def _reshape_embedding(x):
    N, L, D = x.shape
    H = W = int(L ** 0.5)
    x = x.reshape(N, H, W, D)
    x = torch.einsum("nhwd->ndhw", x)
    return x


def _load_vit(model, restore_path):
    # todo: this expects DiT policy ("features") -> needs to match my structure
    if restore_path:
        print("Restoring model from", restore_path)
        state_dict = torch.load(restore_path, map_location="cpu")
        state_dict = (
            state_dict["features"] if "features" in state_dict else state_dict["model"]
        )

        # resize pos_embed if required
        if state_dict["pos_embed"].shape != model.pos_embed.shape:
            print(
                f"resizing pos_embed from {state_dict['pos_embed'].shape} to {model.pos_embed.shape}"
            )
            state_dict["pos_embed"] = resize_pos_embed(
                state_dict["pos_embed"],
                model.pos_embed,
                getattr(model, "num_tokens", 1),
                model.patch_embed.grid_size,
            )

        # filter out keys with name decoder or mask_token
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if "decoder" not in k and "mask_token" not in k
        }

        # remove norm if using global_pool instead of class token
        if model.classifier_feature == "global_pool":
            print("Removing extra weights for global_pool")
            # remove layer that start with norm
            state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith("norm")
            }
            # add fc_norm in the state dict from the model
            state_dict["fc_norm.weight"] = model.fc_norm.weight
            state_dict["fc_norm.bias"] = model.fc_norm.bias

        # load state_dict
        model.load_state_dict(state_dict)
    return model


def _random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(
        noise, dim=1
    )  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore
