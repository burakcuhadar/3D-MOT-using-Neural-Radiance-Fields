import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional

from models.embedder import get_embedder
from models.rendering__ import raw2outputs
from models.resnet import ResnetFC

from typing import Tuple, Optional, Union, TypeAlias

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

RawNerfOutput: TypeAlias = Tuple[
    TensorType["num_rays", "num_samples"],  # raw_alpha
    TensorType["num_rays", "num_samples", 3],  # raw_rgb
]

NerfOutput: TypeAlias = Tuple[
    TensorType["num_rays", 3],  # rgb
    TensorType["num_rays"],  # disp
    TensorType["num_rays"],  # acc
    TensorType["num_rays", "num_samples"],  # weights
    TensorType["num_rays"],  # depth
]


# Model
class NeRF(nn.Module):
    def __init__(self, D, W, args, has_time=False, more_view_layers=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W

        # Create embedders for positional encoding
        if not has_time:
            embedder, input_ch = get_embedder(
                args.multires, args.end_barf, args.i_embed
            )
        else:
            embedder, input_ch = get_embedder(
                args.multires, args.end_barf, args.i_embed, input_dims=4
            )

        input_ch_views = 0
        self.embedder_dirs = None
        if args.use_viewdirs:
            embedder_dirs, input_ch_views = get_embedder(
                args.multires_views, args.end_barf, args.i_embed
            )

        # Used only when not using viewdirs TODO is 5th dim ever used?
        output_ch = 5 if args.N_importance > 0 else 4

        self.embedder = embedder
        self.embedder_dirs = embedder_dirs
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.use_viewdirs = args.use_viewdirs

        self.pts_net = ResnetFC(input_ch, d_out=W, n_blocks=D // 2, d_hidden=W)
        """self.pts_net = tcnn.Network(
            n_input_dims=input_ch, n_output_dims=W,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": W,
                "n_hidden_layers": D,
            }
        )"""

        ### Implementation according to the official code release:
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        if not more_view_layers:
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        else:  # TODO remove if does not work
            self.views_linears = nn.ModuleList(
                [nn.Linear(input_ch_views + W, W // 2)]
                + [nn.Linear(W // 2, W // 2) for i in range(D // 2)]
            )

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if args.use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.netchunk = args.netchunk
        self.raw_noise_std = args.raw_noise_std  # should only be used for training
        self.white_bkgd = args.white_bkgd

        # Weight Initialization
        for layer in self.views_linears:
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            torch.nn.init.zeros_(layer.bias)
        torch.nn.init.kaiming_normal_(self.alpha_linear.weight, nonlinearity="relu")
        torch.nn.init.zeros_(self.alpha_linear.bias)
        torch.nn.init.xavier_uniform_(self.rgb_linear.weight)

    @typechecked
    def forward(
        self,
        pts: TensorType["num_rays", "num_samples", 3],
        viewdirs: TensorType["num_rays", 3],
        z_vals: Optional[TensorType["num_rays", "num_samples"]] = None,
        rays_d: Optional[TensorType["num_rays", 3]] = None,
        step: Optional[int] = None,
        time: Optional[float] = None,
    ) -> Union[RawNerfOutput, NerfOutput]:
        """
        1. Embed the input points and view directions if given
        2. Forward pass embedded inputs through MLP to get rgb and density
        3. Estimate expected color integral
        """
        pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])  # [N_rays*N_samples, 3]

        if time is not None:
            time = torch.ones_like(pts_flat[:, :1]) * time
            pts_flat = torch.cat([pts_flat, time], -1)

        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(pts.shape)
            input_dirs_flat = torch.reshape(
                input_dirs, [-1, input_dirs.shape[-1]]
            )  # [N_rays*N_samples, 3]

        raw_rgb_chunks, raw_alpha_chunks = [], []
        # Forward by chunks to avoid OOM
        for i in range(0, pts_flat.shape[0], self.netchunk):
            end_i = min(pts_flat.shape[0], i + self.netchunk)
            pts_chunk = pts_flat[i:end_i, :]
            # 1. Embed the input points and view directions if given
            embedded_pts = self.embedder(pts_chunk, step=step)

            if viewdirs is not None:
                input_dirs_chunk = input_dirs_flat[i:end_i, :]
                embedded_dirs = self.embedder_dirs(input_dirs_chunk, step=step)

            # 2. Forward pass embedded inputs through MLP to get rgb and density
            h = embedded_pts
            h = self.pts_net(h)

            if self.use_viewdirs:
                raw_alpha = self.alpha_linear(h)
                feature = self.feature_linear(h)
                h = torch.cat([feature, embedded_dirs], -1)

                for i, l in enumerate(self.views_linears):
                    h = self.views_linears[i](h)
                    h = F.relu(h)

                raw_rgb = self.rgb_linear(h)

            else:
                output = self.output_linear(h)
                raw_rgb = output[:, :3]
                raw_alpha = output[:, 3]

            raw_rgb_chunks.append(raw_rgb)
            raw_alpha_chunks.append(raw_alpha)

        raw_rgb = torch.cat(raw_rgb_chunks, dim=0)  # N_rays*N_samples, 3
        raw_alpha = torch.cat(raw_alpha_chunks, dim=0)  # N_rays*N_samples
        raw_rgb = torch.reshape(raw_rgb, pts.shape)  # N_rays, N_samples, 3
        raw_alpha = torch.reshape(raw_alpha, pts.shape[:-1])  # N_rays, N_samples

        if z_vals is None:
            return raw_alpha, raw_rgb

        # 3. Estimate expected color integral and return extras
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw_alpha,
            raw_rgb,
            z_vals,
            rays_d,
            self.raw_noise_std if self.training else 0,
            self.white_bkgd,
        )

        return rgb_map, disp_map, acc_map, weights, depth_map


################################################################################################
# Nerfacc example model (taken from https://github.com/KAIR-BAIR/nerfacc/tree/master/examples) #
################################################################################################


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        output_dim: int = None,  # The number of output tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        hidden_init: Callable = nn.init.xavier_uniform_,
        hidden_activation: Callable = nn.ReLU(),
        output_enabled: bool = True,
        output_init: Optional[Callable] = nn.init.xavier_uniform_,
        output_activation: Optional[Callable] = nn.Identity(),
        bias_enabled: bool = True,
        bias_init: Callable = nn.init.zeros_,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_enabled = output_enabled
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init

        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(
                nn.Linear(in_features, self.net_width, bias=bias_enabled)
            )
            if (self.skip_layer is not None) and (i % self.skip_layer == 0) and (i > 0):
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        if self.output_enabled:
            self.output_layer = nn.Linear(
                in_features, self.output_dim, bias=bias_enabled
            )
        else:
            self.output_dim = in_features

        self.initialize()

    def initialize(self):
        def init_func_hidden(m):
            if isinstance(m, nn.Linear):
                if self.hidden_init is not None:
                    self.hidden_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)

        self.hidden_layers.apply(init_func_hidden)
        if self.output_enabled:

            def init_func_output(m):
                if isinstance(m, nn.Linear):
                    if self.output_init is not None:
                        self.output_init(m.weight)
                    if self.bias_enabled and self.bias_init is not None:
                        self.bias_init(m.bias)

            self.output_layer.apply(init_func_output)

    def forward(self, x):
        inputs = x
        for i in range(self.net_depth):
            x = self.hidden_layers[i](x)
            x = self.hidden_activation(x)
            if (self.skip_layer is not None) and (i % self.skip_layer == 0) and (i > 0):
                x = torch.cat([x, inputs], dim=-1)
        if self.output_enabled:
            x = self.output_layer(x)
            x = self.output_activation(x)
        return x


class DenseLayer(MLP):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            net_depth=0,  # no hidden layers
            **kwargs,
        )


class NerfMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        condition_dim: int,  # The number of condition tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
    ):
        super().__init__()
        self.base = MLP(
            input_dim=input_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            output_enabled=False,
        )
        hidden_features = self.base.output_dim
        self.sigma_layer = DenseLayer(hidden_features, 1)

        if condition_dim > 0:
            self.bottleneck_layer = DenseLayer(hidden_features, net_width)
            self.rgb_layer = MLP(
                input_dim=net_width + condition_dim,
                output_dim=3,
                net_depth=net_depth_condition,
                net_width=net_width_condition,
                skip_layer=None,
            )
        else:
            self.rgb_layer = DenseLayer(hidden_features, 3)

    def query_density(self, x):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        return raw_sigma

    def forward(self, x, condition=None):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        if condition is not None:
            if condition.shape[:-1] != x.shape[:-1]:
                num_rays, n_dim = condition.shape
                condition = condition.view(
                    [num_rays] + [1] * (x.dim() - condition.dim()) + [n_dim]
                ).expand(list(x.shape[:-1]) + [n_dim])
            bottleneck = self.bottleneck_layer(x)
            x = torch.cat([bottleneck, condition], dim=-1)
        raw_rgb = self.rgb_layer(x)
        return raw_rgb, raw_sigma


class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (int(self.use_identity) + (self.max_deg - self.min_deg) * 2) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent


class VanillaNeRFRadianceField(nn.Module):
    def __init__(
        self,
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
    ) -> None:
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 10, True)
        self.view_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.mlp = NerfMLP(
            input_dim=self.posi_encoder.latent_dim,
            condition_dim=self.view_encoder.latent_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            net_depth_condition=net_depth_condition,
            net_width_condition=net_width_condition,
        )

    def query_opacity(self, x, step_size):
        density = self.query_density(x)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    def query_density(self, x):
        x = self.posi_encoder(x)
        sigma = self.mlp.query_density(x)
        return F.relu(sigma)

    def forward(self, x, condition=None):
        x = self.posi_encoder(x)
        if condition is not None:
            condition = self.view_encoder(condition)
        rgb, sigma = self.mlp(x, condition=condition)
        return torch.sigmoid(rgb), F.relu(sigma)
