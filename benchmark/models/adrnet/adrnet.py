"""ADRnet backbone (vendored from ``deepADRnet/adrnetwork.py``) + ACT-NeuralOP wrapper.

The Advection-Diffusion-Reaction network. Adapted from the original in two ways so it
fits the ACT-NeuralOP pipeline:

* ``color_preserving_advection`` builds its base grid lazily on the input's device/size
  instead of a hard-coded ``cuda`` grid fixed at construction time.
* ``ADRNet`` wraps ``ADRnetCore`` to the :class:`~benchmark.models.common.BaseModel`
  contract (``dim_in``/``dim_out``/``n_spatial_dims``/``spatial_resolution`` injected) and
  feeds a zero "time" input (this ns setup has no physical time-conditioning), returning a
  plain channels-first tensor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.models.common import BaseModel


def CLP(dim_in, dim_out, shape=[256, 256], kernel_size=[3, 3]):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=kernel_size[0] // 2),
        nn.LayerNorm(shape),
        nn.SiLU(),
        nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, padding=kernel_size[0] // 2),
    )


class color_preserving_advection(nn.Module):
    """Color-conserving push-forward advection via ``grid_sample``.

    The base sampling grid is built lazily and cached per (device, H, W) so the module
    works on CPU or GPU without a construction-time ``device`` argument.
    """

    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self._grid_key = None
        self._grid = None

    def _base_grid(self, ref: torch.Tensor) -> torch.Tensor:
        grid_h, grid_w = self.shape[0], self.shape[1]
        key = (ref.device, grid_h, grid_w)
        if self._grid_key != key or self._grid is None:
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, grid_h, device=ref.device),
                torch.linspace(-1, 1, grid_w, device=ref.device),
                indexing="ij",
            )
            self._grid = torch.stack((x, y), dim=-1).unsqueeze(0).unsqueeze(0)
            self._grid_key = key
        return self._grid

    def forward(self, T, U, V):
        UV = torch.stack((U, V), dim=-1)
        transformation_grid = self._base_grid(T) + UV
        Th = F.grid_sample(T, transformation_grid.squeeze(1), align_corners=True)
        return Th


class advectionColorBlock(nn.Module):
    def __init__(self, c, mesh_size):
        super().__init__()
        self.Adv = color_preserving_advection(mesh_size)

        self.ConvU1 = nn.Conv2d(c, c, kernel_size=[3, 3], padding=1)
        self.LNU1 = nn.LayerNorm(normalized_shape=mesh_size)
        self.ConvU2 = nn.Conv2d(c, c, kernel_size=[3, 3], padding=1, bias=False)
        self.ConvU2.weight = nn.Parameter(1e-4 * torch.randn(c, c, 3, 3))

        self.TimeEmbedU = nn.Parameter(1e-4 * torch.randn(1, c, mesh_size[0], mesh_size[1]))
        self.TNU = CLP(c, c, mesh_size)
        self.ConvV1 = nn.Conv2d(c, c, kernel_size=[3, 3], padding=1)
        self.LNV1 = nn.LayerNorm(normalized_shape=mesh_size)
        self.ConvV2 = nn.Conv2d(c, c, kernel_size=[3, 3], padding=1, bias=False)
        self.ConvV2.weight = nn.Parameter(1e-4 * torch.randn(c, c, 3, 3))

        self.TimeEmbedV = nn.Parameter(1e-4 * torch.randn(1, c, mesh_size[0], mesh_size[1]))
        self.TNV = CLP(c, c, mesh_size)

    def forward(self, x, t):
        nw, nh = x.shape[2], x.shape[3]
        teU = t.reshape([-1, 1, 1, 1]) * self.TimeEmbedU
        teU = self.TNU(teU)
        teV = t.reshape([-1, 1, 1, 1]) * self.TimeEmbedV
        teV = self.TNV(teV)

        U = self.ConvU1(x) + teU
        U = self.LNU1(U)
        U = F.silu(U)
        U = self.ConvU2(U)

        V = self.ConvV1(x) + teV
        V = self.LNV1(V)
        V = F.silu(V)
        V = self.ConvV2(V)
        U, V = U / nw, V / nw

        xr = x.reshape(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3])
        Ur = U.reshape(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3])
        Vr = V.reshape(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3])

        xr = self.Adv(xr, Ur, Vr)
        x = xr.reshape(x.shape)
        return x


class ADRnetCore(nn.Module):
    """Advection-Augmented Convolutional Neural Network (original ``ADRnet``)."""

    def __init__(self, in_c, hid_c, out_c, nlayers=16, imsz=[256, 256]):
        super().__init__()
        self.nlayers = nlayers
        self.Open = CLP(in_c, hid_c, imsz)

        self.Adv = nn.ParameterList()
        self.DR = nn.ParameterList()
        for _ in range(nlayers):
            self.Adv.append(advectionColorBlock(hid_c, imsz))
            self.DR.append(CLP(hid_c, hid_c, imsz, kernel_size=[5, 5]))

        self.Close = nn.Parameter(torch.randn(out_c, hid_c, 1, 1) * 1e-2)
        self.h = 1 / imsz[0]

    def forward(self, x, t):
        z = self.Open(x)
        for i in range(self.nlayers):
            dz = self.Adv[i](z, t)
            dz = self.DR[i](dz)
            z = z + self.h * dz
        x = F.conv2d(z, self.Close)
        return x


class ADRNet(BaseModel):
    """ACT-NeuralOP wrapper around :class:`ADRnetCore`.

    Matches the model-instantiation contract in ``benchmark/train.py``. Since this ns
    setup carries no physical time-conditioning, a zero time tensor is fed to the
    advection blocks. Returns a plain channels-first tensor prediction.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
        hidden_channels: int = 32,
        nlayers: int = 16,
        gradient_checkpointing: bool = False,
    ):
        super().__init__(n_spatial_dims, spatial_resolution)
        if n_spatial_dims != 2:
            raise ValueError(f"ADRNet expects n_spatial_dims==2, got {n_spatial_dims}")
        imsz = [int(spatial_resolution[0]), int(spatial_resolution[1])]
        self.net = ADRnetCore(
            in_c=dim_in,
            hid_c=hidden_channels,
            out_c=dim_out,
            nlayers=nlayers,
            imsz=imsz,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.new_zeros(x.shape[0])
        return self.net(x, t)
