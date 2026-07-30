"""
Mixed adaptation from:

    Liu et al. 2022, A ConvNet for the 2020s.
    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Ronneberger et al., 2015. Convolutional Networks for Biomedical Image Segmentation.

If you use this implementation, please cite original work above.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
from torch.utils.checkpoint import checkpoint

from benchmark.models.common import BaseModel
from benchmark.models.act_fno import MultiHeadACTBlock
from benchmark.models.act_fno3d import MultiHeadACTBlock3d
from benchmark.models.deform_bounds import max_displacement_normalized_one_cell

conv_modules = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
conv_transpose_modules = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}

permute_channel_strings = {
    2: [
        "N C H W -> N H W C",
        "N H W C -> N C H W",
    ],
    3: [
        "N C D H W -> N D H W C",
        "N D H W C -> N C D H W",
    ],
}


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self, normalized_shape, n_spatial_dims, eps=1e-6, data_format="channels_last"
    ):
        super().__init__()
        if data_format == "channels_last":
            padded_shape = (normalized_shape,)
        else:
            padded_shape = (normalized_shape,) + (1,) * n_spatial_dims
        self.weight = nn.Parameter(torch.ones(padded_shape))
        self.bias = nn.Parameter(torch.zeros(padded_shape))
        self.n_spatial_dims = n_spatial_dims
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            x = F.normalize(x, p=2, dim=1, eps=self.eps) * self.weight
            return x


class Upsample(nn.Module):
    r"""Upsample layer."""

    def __init__(self, dim_in, dim_out, n_spatial_dims=2):
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm(dim_in, n_spatial_dims, eps=1e-6, data_format="channels_first"),
            conv_transpose_modules[n_spatial_dims](
                dim_in, dim_out, kernel_size=2, stride=2
            ),
        )

    def forward(self, x):
        return self.block(x)


class Downsample(nn.Module):
    r"""Downsample layer."""

    def __init__(self, dim_in, dim_out, n_spatial_dims=2):
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm(dim_in, n_spatial_dims, eps=1e-6, data_format="channels_first"),
            conv_modules[n_spatial_dims](dim_in, dim_out, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, n_spatial_dims, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.dwconv = conv_modules[n_spatial_dims](
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, n_spatial_dims, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # (N, C, H, W) -> (N, H, W, C)
        x = rearrange(x, permute_channel_strings[self.n_spatial_dims][0])
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # (N, H, W, C) -> (N, C, H, W)
        x = rearrange(x, permute_channel_strings[self.n_spatial_dims][1])
        x = input + self.drop_path(x)
        return x


def _make_deform_block_unet_convnext(
    n_spatial_dims: int,
    channels: int,
    spatial_resolution: tuple[int, ...],
    num_heads: int,
    disp_net_branched: bool,
    max_disp: float | None,
    deform_hidden_dim: int,
    skip_output_norm: bool = False,
) -> nn.Module:
    if max_disp is not None:
        deform_max = float(max_disp)
    else:
        deform_max = max_displacement_normalized_one_cell(spatial_resolution)
    if n_spatial_dims == 2:
        return MultiHeadACTBlock(
            channels,
            channels,
            num_heads=num_heads,
            hidden_dim=deform_hidden_dim,
            max_disp=deform_max,
            use_norm=True,
            disp_net_branched=disp_net_branched,
            skip_output_norm=skip_output_norm,
        )
    if n_spatial_dims == 3:
        return MultiHeadACTBlock3d(
            channels,
            channels,
            num_heads=num_heads,
            hidden_dim=deform_hidden_dim,
            max_disp=deform_max,
            use_norm=True,
            disp_net_branched=disp_net_branched,
            skip_output_norm=skip_output_norm,
        )
    raise ValueError(
        f"ACT blocks require n_spatial_dims in (2, 3); got {n_spatial_dims}."
    )


class Stage(nn.Module):
    r"""ConvNeXt Stage.
    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        n_spatial_dims (int): Number of spatial dimensions.
        depth (int): Number of blocks in the stage.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        mode (str): Down, Up, Neck. Default: "down"
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        n_spatial_dims,
        depth=1,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        mode="down",
        skip_project=False,
    ):
        super().__init__()

        if skip_project:
            self.skip_proj = conv_modules[n_spatial_dims](2 * dim_in, dim_in, 1)
        else:
            self.skip_proj = nn.Identity()
        if mode == "down":
            self.resample = Downsample(dim_in, dim_out, n_spatial_dims)
        elif mode == "up":
            self.resample = Upsample(dim_in, dim_out, n_spatial_dims)
        else:
            self.resample = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                Block(dim_in, n_spatial_dims, drop_path, layer_scale_init_value)
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        x = self.skip_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.resample(x)
        return x


class EncoderStagePostACT(nn.Module):
    r"""Encoder :class:`Stage` (ConvNeXt blocks + downsample), then a :class:`MultiHeadACTBlock`."""

    def __init__(
        self,
        dim_in,
        dim_out,
        n_spatial_dims,
        spatial_resolution: tuple[int, ...],
        depth=1,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        num_heads: int = 16,
        disp_net_branched: bool = True,
        max_disp: float | None = None,
        deform_hidden_dim: int = 16,
        skip_output_norm: bool = False,
    ):
        super().__init__()
        self.stage = Stage(
            dim_in,
            dim_out,
            n_spatial_dims,
            depth,
            drop_path,
            layer_scale_init_value,
            mode="down",
        )
        self.deform = _make_deform_block_unet_convnext(
            n_spatial_dims,
            dim_out,
            spatial_resolution,
            num_heads,
            disp_net_branched,
            max_disp,
            deform_hidden_dim,
            skip_output_norm=skip_output_norm,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list]:
        x = self.stage(x)
        x, d = self.deform(x)
        disps = [d] if d is not None else []
        return x, disps


class DecoderStagePostACT(nn.Module):
    r"""Decoder :class:`Stage` (optional skip, ConvNeXt blocks + upsample), then a :class:`MultiHeadACTBlock`."""

    def __init__(
        self,
        dim_in,
        dim_out,
        n_spatial_dims,
        spatial_resolution: tuple[int, ...],
        depth=1,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        skip_project=False,
        num_heads: int = 16,
        disp_net_branched: bool = True,
        max_disp: float | None = None,
        deform_hidden_dim: int = 16,
        skip_output_norm: bool = False,
    ):
        super().__init__()
        self.stage = Stage(
            dim_in,
            dim_out,
            n_spatial_dims,
            depth,
            drop_path,
            layer_scale_init_value,
            mode="up",
            skip_project=skip_project,
        )
        self.deform = _make_deform_block_unet_convnext(
            n_spatial_dims,
            dim_out,
            spatial_resolution,
            num_heads,
            disp_net_branched,
            max_disp,
            deform_hidden_dim,
            skip_output_norm=skip_output_norm,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list]:
        x = self.stage(x)
        x, d = self.deform(x)
        disps = [d] if d is not None else []
        return x, disps


class UNet_ConvNext(BaseModel):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
        stages: int = 4,
        blocks_per_stage: int = 1,
        blocks_at_neck: int = 1,
        init_features: int = 32,
        gradient_checkpointing: bool = False,
    ):
        super().__init__(n_spatial_dims, spatial_resolution)
        self.n_spatial_dims = n_spatial_dims
        features = init_features
        self.gradient_checkpointing = gradient_checkpointing
        encoder_dims = [features * 2**i for i in range(stages + 1)]
        decoder_dims = [features * 2**i for i in range(stages, -1, -1)]
        encoder = []
        decoder = []
        self.in_proj = conv_modules[n_spatial_dims](
            dim_in, features, kernel_size=3, padding=1
        )
        self.out_proj = conv_modules[n_spatial_dims](
            features, dim_out, kernel_size=3, padding=1
        )
        for i in range(stages):
            encoder.append(
                Stage(
                    encoder_dims[i],
                    encoder_dims[i + 1],
                    n_spatial_dims,
                    blocks_per_stage,
                    mode="down",
                )
            )
            decoder.append(
                Stage(
                    decoder_dims[i],
                    decoder_dims[i + 1],
                    n_spatial_dims,
                    blocks_per_stage,
                    mode="up",
                    skip_project=i != 0,
                )
            )
        self.encoder = nn.ModuleList(encoder)
        self.neck = Stage(
            encoder_dims[-1],
            encoder_dims[-1],
            n_spatial_dims,
            blocks_at_neck,
            mode="neck",
        )
        self.decoder = nn.ModuleList(decoder)

    def optional_checkpointing(self, layer, *inputs, **kwargs):
        if self.gradient_checkpointing:
            return checkpoint(layer, *inputs, use_reentrant=False, **kwargs)
        else:
            return layer(*inputs, **kwargs)

    def forward(self, x):
        x = self.in_proj(x)
        skips = []
        for i, enc in enumerate(self.encoder):
            skips.append(x)
            x = self.optional_checkpointing(enc, x)
        x = self.neck(x)
        for j, dec in enumerate(self.decoder):
            if j > 0:
                x = torch.cat([x, skips[-j]], dim=1)
            x = dec(x)
        x = self.out_proj(x)
        return x


class ACTUNetConvNext(BaseModel):
    """U-Net + ConvNeXt with one multi-head ACT block **after** each encoder and decoder :class:`Stage`.

    Each stage is ConvNeXt blocks then downsample (encoder) or upsample (decoder); ACT runs
    on the stage output resolution and channel width. The neck remains a plain :class:`Stage`.
    Forward returns ``(prediction, disps)`` like :class:`~benchmark.models.act_fno.ACTFNO`.
    Only ``n_spatial_dims`` in ``{2, 3}`` is supported.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
        stages: int = 4,
        blocks_per_stage: int = 1,
        blocks_at_neck: int = 1,
        init_features: int = 32,
        gradient_checkpointing: bool = False,
        num_heads: int = 4,
        disp_net_branched: bool = True,
        max_disp: float | None = None,
        deform_hidden_dim: int = 64,
    ):
        super().__init__(n_spatial_dims, spatial_resolution)
        if n_spatial_dims not in (2, 3):
            raise ValueError(
                f"ACTUNetConvNext expects n_spatial_dims in (2, 3); got {n_spatial_dims}"
            )
        self.n_spatial_dims = n_spatial_dims
        features = init_features
        self.gradient_checkpointing = gradient_checkpointing
        encoder_dims = [features * 2**i for i in range(stages + 1)]
        decoder_dims = [features * 2**i for i in range(stages, -1, -1)]
        encoder = []
        decoder = []
        self.in_proj = conv_modules[n_spatial_dims](
            dim_in, features, kernel_size=3, padding=1
        )
        self.out_proj = conv_modules[n_spatial_dims](
            features, dim_out, kernel_size=3, padding=1
        )
        for i in range(stages):
            encoder.append(
                EncoderStagePostACT(
                    encoder_dims[i],
                    encoder_dims[i + 1],
                    n_spatial_dims,
                    spatial_resolution,
                    blocks_per_stage,
                    num_heads=num_heads,
                    disp_net_branched=disp_net_branched,
                    max_disp=max_disp,
                    deform_hidden_dim=deform_hidden_dim,
                    skip_output_norm=False,
                )
            )
            is_last_dec = i == stages - 1
            decoder.append(
                DecoderStagePostACT(
                    decoder_dims[i],
                    decoder_dims[i + 1],
                    n_spatial_dims,
                    spatial_resolution,
                    blocks_per_stage,
                    skip_project=i != 0,
                    num_heads=num_heads,
                    disp_net_branched=disp_net_branched,
                    max_disp=max_disp,
                    deform_hidden_dim=deform_hidden_dim,
                    skip_output_norm=is_last_dec,
                )
            )
        self.encoder = nn.ModuleList(encoder)
        self.neck = Stage(
            encoder_dims[-1],
            encoder_dims[-1],
            n_spatial_dims,
            blocks_at_neck,
            mode="neck",
        )
        self.decoder = nn.ModuleList(decoder)

    def optional_checkpointing(self, layer, *inputs, **kwargs):
        if self.gradient_checkpointing:
            return checkpoint(layer, *inputs, use_reentrant=False, **kwargs)
        return layer(*inputs, **kwargs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        all_disps: list[torch.Tensor] = []
        x = self.in_proj(x)
        skips = []
        for enc in self.encoder:
            skips.append(x)
            out = self.optional_checkpointing(enc, x)
            if isinstance(out, tuple):
                x, ds = out
                all_disps.extend(ds)
            else:
                x = out
        x = self.optional_checkpointing(self.neck, x)
        for j, dec in enumerate(self.decoder):
            if j > 0:
                x = torch.cat([x, skips[-j]], dim=1)
            out = self.optional_checkpointing(dec, x)
            if isinstance(out, tuple):
                x, ds = out
                all_disps.extend(ds)
            else:
                x = out
        x = self.out_proj(x)
        return x, all_disps
