# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# adapted from:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
# Modified by Sudeep Dasari
# Modified by Jannick Stranghöner

from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
from timm.models.vision_transformer import resize_pos_embed

from coro.common.model.base_nets import BaseEncoder
from coro.common.utils.obs_utils import register_encoder

"""
================================================
ViT Factories
================================================
"""


@register_encoder(modality='rgb')
def vit_small_patch16(input_shape, **kwargs):
    """ViT small as defined in the DeiT paper."""
    model = VisionTransformer(
        input_shape,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


@register_encoder(modality='rgb')
def vit_base_patch16(input_shape, **kwargs):
    model = VisionTransformer(
        input_shape,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


@register_encoder(modality='rgb')
def clip_vit_base_patch16(input_shape, **kwargs):
    model = ClipVisionTransformer(
        input_shape,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # CLIP-specific:
        pre_norm=True,
        num_classes=512,
        **kwargs,
    )
    return model


@register_encoder(modality='rgb')
def vit_large_patch16(input_shape, **kwargs):
    model = VisionTransformer(
        input_shape,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


"""
================================================
ViT Wrappers
================================================
"""


@register_encoder(
    modality='rgb',
    name='vit_huge_patch14',
    patch_size=14,
    embed_dim=1280,
    depth=32,
    num_heads=16,
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6)
)
class VisionTransformer(BaseEncoder):
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
            img_size=224,
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
        model = timm.models.vision_transformer.VisionTransformer(img_size=img_size, patch_size=patch_size,
                                                                 embed_dim=embed_dim, depth=depth,
                                                                 num_heads=num_heads, mlp_ratio=mlp_ratio,
                                                                 qkv_bias=qkv_bias,
                                                                 norm_layer=norm_layer, drop_path_rate=drop_path_rate,
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
        self._model = load_vit(self._model, restore_path)
        self._embed_dim = embed_dim

    def random_masking(self, x, mask_ratio):
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

    def handle_outcome(self, x):
        if self._model.classifier_feature == "global_pool":
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self._model.fc_norm(x)
        elif self._model.classifier_feature == "use_cls_token":
            x = self._model.norm(x)
            outcome = x[:, :1]  # use cls token
        elif self._model.classifier_feature == "reshape_embedding":
            x = self._model.norm(x)
            outcome = reshape_embedding(
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
            x, _, _ = self.random_masking(x, mask_ratio=self.mask_ratio)

        # append cls token
        cls_token = self._model.cls_token + self._model.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(B, -1, -1), x), dim=1)

        x = self._model.blocks(x)
        return self.handle_outcome(x)

    def forward(self, x):
        return self.forward_features(x)

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
            return [self._model.embed_dim, 1]
        elif self._model.classifier_feature == "global_pool":
            return [self._model.embed_dim, num_patches]
        else:
            raise NotImplementedError("Unknown classifier feature type.")


# todo: this implementation needs to be evaluated
@register_encoder()
class ClipVisionTransformer(VisionTransformer):
    def forward_features(self, x):
        B = x.shape[0]
        x = self._model.patch_embed(x)
        x = torch.cat(
            [
                self._model.cls_token.squeeze()
                + torch.zeros(B, 1, x.shape[-1], device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self._model.pos_embed.squeeze().to(x.dtype)
        x = self._model.norm_pre(x)

        x = self._model.blocks(x)
        return self.handle_outcome(x)


def reshape_embedding(x):
    N, L, D = x.shape
    H = W = int(L ** 0.5)
    x = x.reshape(N, H, W, D)
    x = torch.einsum("nhwd->ndhw", x)
    return x


def load_vit(model, restore_path):
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


if __name__ == '__main__':
    # Define input tensor (random image batch with batch_size=1, channels=3, height=224, width=224)
    input_tensor = torch.randn(1, 3, 224, 224)

    # Instantiate ClipVisionTransformer and VisionTransformer models using the factory functions
    clip_vit_model = clip_vit_base_patch16()
    vit_model = vit_base_patch16()

    # Test the forward pass for the ClipVisionTransformer
    print("Testing ClipVisionTransformer...")
    clip_output = clip_vit_model(input_tensor)
    print(f"ClipVisionTransformer output shape: {clip_output.shape}")

    # Check output shape from the output_shape function
    clip_output_shape = clip_vit_model.output_shape(input_tensor.shape)
    print(f"ClipVisionTransformer output shape from method: {clip_output_shape}")

    # Test the forward pass for the VisionTransformer
    print("Testing VisionTransformer...")
    vit_output = vit_model(input_tensor)
    print(f"VisionTransformer output shape: {vit_output.shape}")

    # Check output shape from the output_shape function
    vit_output_shape = vit_model.output_shape(input_tensor.shape)
    print(f"VisionTransformer output shape from method: {vit_output_shape}")

    # Optional: Test with more ViT variations like vit_small_patch16(), vit_large_patch16(), etc.
    print("Testing small and large variants...")

    vit_small = vit_small_patch16()
    vit_large = vit_large_patch16()

    small_output = vit_small(input_tensor)
    large_output = vit_large(input_tensor)

    print(f"Small ViT output shape: {small_output.shape}")
    print(f"Large ViT output shape: {large_output.shape}")
