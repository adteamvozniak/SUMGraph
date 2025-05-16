import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, GATConv, global_max_pool, global_add_pool
from torch_geometric.data import Batch 
from torch_geometric.nn import GraphMultisetTransformer

from torch_geometric.nn import GlobalAttention
from torch_geometric.nn import Set2Set
from torchvision.ops import RoIAlign

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def modulate(x, shift, scale):
    # Adjust the shape of shift and scale to match the input tensor
    # Shift and scale are expanded from [batch_size, 1] to [batch_size, 1, 1, 1]
    # to match the input tensor shape [batch_size, height, width, channels]
    shift = shift.view(shift.size(0), 1, 1, 1)
    scale = scale.view(scale.size(0), 1, 1, 1)

    return x * (1 + scale) + shift


def modulate_scale(x, scale):
    scale = scale.view(scale.size(0), 1, 1, 1)
    return x * (1 + scale)


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        #self.graph_hidden_dim = graph_hidden_dim  # ✅ Store graph hidden size

        ## 🔥 Projection to align graph and visual feature dimensions
        #self.graph_projection = nn.Linear(self.graph_hidden_dim, self.d_model)  

        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        #self.graph_projection = nn.Linear(graph_hidden_dim, self.d_model)

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor,  modulation_params: torch.Tensor = None, **kwargs):
        if modulation_params is not None:
            # Unpack modulation parameters
            shift1, scale1, shift2, scale2, scale3 = torch.chunk(modulation_params, 5, dim=-1)

        B, H, W, C = x.shape

        # Apply modulation after layer norm 1
        if modulation_params is not None:
            x = modulate(x, shift1, scale1)

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        # Apply modulation after SS2D block (only scale)
        if modulation_params is not None:
            y = modulate_scale(y, scale3)

        y = self.out_norm(y)

        # Apply modulation after last layer norm
        if modulation_params is not None:
            y = modulate(y, shift2, scale2)

        y = y * F.silu(z)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

    '''def forward(self, x: torch.Tensor, graph_features: torch.Tensor = None, modulation_params: torch.Tensor = None, **kwargs):
        if modulation_params is not None:
            # Unpack modulation parameters
            shift1, scale1, shift2, scale2, scale3 = torch.chunk(modulation_params, 5, dim=-1)

        B, H, W, C = x.shape

        # 🔥 Inject Graph Features
        if graph_features is not None:
            graph_features = self.graph_projection(graph_features)  # Align dimensions        
            graph_features = graph_features.unsqueeze(1).unsqueeze(1)  # Expand to (B, 1, 1, C)
            graph_features = graph_features.expand(-1, H, W, -1)  # Match spatial size

            x = x + graph_features  # Fuse graph-based attention

        # Apply modulation after layer norm 1
        if modulation_params is not None:
            x = modulate(x, shift1, scale1)

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        # Apply modulation after SS2D block (only scale)
        if modulation_params is not None:
            y = modulate_scale(y, scale3)

        y = self.out_norm(y)

        # Apply modulation after last layer norm
        if modulation_params is not None:
            y = modulate(y, shift2, scale2)

        y = y * F.silu(z)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out'''

'''
class VSSBlock_graph(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            condition_size: int = 4,
            use_modulation: bool = False,
            graph_input_dim: int =  26,   #v4 56, #v1-v3  28,  # Input dimension for graph node features
            graph_hidden_dim: int = 26, #v4 56, #v1-v3 28, # Hidden dimension for graph processing
            attr_dim: int = 3,          # Dimension of vehicle attributes (e.g., speed, color, etc.)
            use_attributes: bool = True, # use global attributes
            **kwargs,
    ):
        super().__init__()
        num_class = 5
        self.use_modulation = use_modulation
        self.ln_1 = norm_layer(hidden_dim)
        self.use_attributes = use_attributes
        
        self.graph_input_dim = graph_input_dim

        self.num_representatives = 5

        #self.gmt_pooling = GraphMultisetTransformer(
        #    channels=graph_hidden_dim,  # Feature size of input
        #    k=self.num_representatives,  # Number of representative nodes after pooling
        #    num_encoder_blocks=1,  # Number of SAB blocks between pooling
        #    heads=2,  # Multi-head attention
        #    layer_norm=True,  # Apply Layer Norm
        #    dropout=0.01  # Dropout probability for attention weights
        #)

        #self.att_gate = nn.Linear(graph_hidden_dim, 1)  # Learnable attention gate
        #self.global_attention_pooling = GlobalAttention(self.att_gate)

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  
            nn.Sigmoid()
        )

        # Output layer (optional, if final output needs modification)
        #self.fc_out = nn.Linear(hidden_dim, output_dim)

        #v4
        # ✅ Dynamically infer input feature channels
        #self.input_feature_dim = None  # Will be set dynamically in forward()
        #v4
        # Projection layer (will initialize later)
        #self.visual_feature_projection = None  

        #self.fusion_layer = nn.Linear(graph_hidden_dim + attr_dim, graph_hidden_dim)  # Reduce back to graph_hidden_dim

        # Validate hidden_dim and d_state
        if d_state <= 0 or hidden_dim <= 0:
            raise ValueError(f"Invalid d_state ({d_state}) or hidden_dim ({hidden_dim}). Both must be positive.")
        if hidden_dim % d_state != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by d_state ({d_state}).")
        

        # Self-attention layer for visual processing
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

        # Graph processing layers for vehicle skeletons
        self.conv1 = GCNConv(graph_input_dim, graph_hidden_dim)
        self.conv2 = GCNConv(graph_hidden_dim, graph_hidden_dim)
        
        #v2
        #self.conv1_att = GATConv(graph_input_dim, graph_hidden_dim , heads=4, concat=True)
        #self.conv2_att = GATConv(graph_hidden_dim*4, graph_hidden_dim, heads=1, concat=False)
        #v4
        #self.fusion_gate = nn.Linear(graph_hidden_dim, hidden_dim)  # Projects concatenated features

        # Linear layer to process vehicle attributes
        # v1-v2
        self.attr_fc = nn.Linear(attr_dim, graph_hidden_dim)
        # v3 model
        #self.attr_fc = nn.Linear(attr_dim, graph_hidden_dim*3)
        # Projection layer to align graph and visual feature dimensions
        self.graph_fc = nn.Linear(graph_hidden_dim, hidden_dim)
        #self.set_transformer_pooling = Set2Set(graph_hidden_dim, processing_steps=3)
        #self.graph_fc_avg_pooling = nn.Linear(graph_hidden_dim*3, graph_hidden_dim)
        #nn.Linear(288, 96)
        #v4
        # RoI Align for extracting CNN features at vehicle positions
        #self.roi_align = RoIAlign(output_size=1, spatial_scale=1.0, sampling_ratio=-1)
        #self.cnn_feature_projection = nn.Linear(cnn_feature_dim, graph_input_dim)  # ✅ Projection to align dimensions
        # ✅ Projection layer to align bilinear-sampled image features with graph nodes
        
        # Modulation network, if enabled
        if self.use_modulation:
            self.condition = nn.Parameter(torch.randn(condition_size, 128), requires_grad=True)
            self.mlp = nn.Sequential(
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 5)
            )

    #v3 model 
    
    def forward(self, input: torch.Tensor, condition: torch.Tensor, vehicles_graphs: Batch, attributes: torch.Tensor):
        """
        Forward pass of the extended VSSBlock with batched vehicle skeleton graphs and attributes.

        Parameters:
        - input: Visual feature tensor (e.g., from an image).
        - condition: Condition tensor for modulation.
        - vehicles_graphs: Batched vehicle graphs, where each vehicle is annotated with COCO-format skeleton (24 vertices).
        - attributes: Attribute tensor for each vehicle (e.g., speed, color), shape (num_vehicles, attr_dim).
        """
        
        # ✅ Skip GCN processing if the graph is empty
        if vehicles_graphs.x.shape[0] == 0:
            #print("⚠️ No vehicle graphs available. Skipping GCN processing.")
            # Step 4: Conditional Modulation (if enabled)
            selected_modulation = None
            if self.use_modulation:
                modulation_params = self.mlp(self.condition)
                if condition.dim() == 1:
                    condition = F.one_hot(condition, num_classes=self.condition.size(0)).float()
                indices = torch.argmax(condition, dim=1)
                selected_modulation = torch.stack([modulation_params[idx] for idx in indices])

            # Step 5: Self-attention and Modulation
            x = input + self.drop_path(self.self_attention(self.ln_1(input), modulation_params=selected_modulation))
            return x

        # Step 1: Process vehicle skeleton graphs in batch
        # Extract Graph Features
        x, edge_index, pos, batch = (
            vehicles_graphs.x, 
            vehicles_graphs.edge_index, 
            vehicles_graphs.pos, 
            vehicles_graphs.batch  # ✅ Extract `batch` for global_mean_pool
        )

        #else:
        #    print("✅ Vehicle graphs available. Involving GCN processing.")

        #if pos is not None and pos.shape[0] == x.shape[0]:
        #    x = torch.cat([x, pos], dim=1)  # Append position data to node features


        # Apply GCN layers to each vehicle skeleton
        # v1 and v2 models
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # v3 of a model with using graph attention nn
        #x = F.relu(self.conv1_att(x, edge_index))
        #x = F.relu(self.conv2_att(x, edge_index))

        # v1 and v2 models
        # Aggregate features across all vertices for each vehicle
        vehicle_embeddings = global_mean_pool(x, batch)  # Aggregate node features per graph/vehicle

        #v3 model
        #vehicle_embeddings = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch), global_add_pool(x, batch)], dim=-1)

        #vehicle_embeddings = self.graph_fc_avg_pooling(vehicle_embeddings)

        #vehicle_embeddings = self.gmt_pooling(x, batch)

        #print(f"batch.max() + 1 = {batch.max().item() + 1}")
        #print(f"attributes.shape[0] = {attributes.shape[0]}")

        #vehicle_embeddings = self.set_transformer_pooling(x, batch)  # Set transformer pooling
        # final training
        
        if self.use_attributes and attributes is not None:
            # Step 3: Ensure attributes match the number of graphs (trim if necessary)

            #print("We skip attributes")

            #num_graphs = torch.unique(batch).numel()
            num_graphs = batch.max().item() + 1  # Get the actual number of graphs
            
            #print(f"Before Trimming: Attributes Shape: {attributes.shape}, Num Graphs (Computed): {num_graphs}")
            #print(f"Batch Tensor: {batch}, Shape: {batch.shape}, Max: {batch.max().item()}")
            
            if attributes.shape[0] > num_graphs:
                #print(f"⚠ Mismatched attribute size! Trimming from {attributes.shape[0]} to {num_graphs}")
                attributes = attributes[:num_graphs]  # Trim extra attributes

            # Step 2: Process attributes and combine with graph features
            attributes_emb = F.relu(self.attr_fc(attributes))  # Transform attributes to match graph feature dimension
            # v1 and v2 models
            vehicle_features = vehicle_embeddings + attributes_emb  # Combine node and attribute features
            #v2
            #vehicle_features = torch.cat([vehicle_embeddings, attributes_emb], dim=-1)  # Shape: (batch_size, graph_hidden_dim + attr_dim)
            #vehicle_features = self.fusion_layer(vehicle_features)  # Apply a learnable transformation
        else:
            vehicle_features = vehicle_embeddings

        # Step 3: Pool vehicle-level embeddings to get a scene-level context
        scene_context = vehicle_features.mean(dim=0, keepdim=True)  # Overall context for all vehicles in scene

        # Project the scene context to match the visual input dimension
        scene_context = self.graph_fc(scene_context)
        
        #print(f"Scene Context Shape: {scene_context.shape}")  

        #fusion_weight = torch.sigmoid(self.fusion_gate(scene_context))  # Compute adaptive fusion weight
        #print(f"Scene Weights Shape: {fusion_weight.shape}")  # Expect (26, 96)
        #input = input + fusion_weight * scene_context  # Weighted fusion

        input = input + scene_context  # Fuse scene context with visual features  <-- is this one wrong?    v1 model
        

        #input2 = input + scene_context2  #v2 model
        #print(f"scene_content1.shape={scene_context1.shape}")
        #print(f"scene_content2.shape={scene_context2.shape}")



        # the result should be called input2

        # Step 4: Conditional Modulation (if enabled)
        selected_modulation = None
        if self.use_modulation:
            modulation_params = self.mlp(self.condition)
            if condition.dim() == 1:
                condition = F.one_hot(condition, num_classes=self.condition.size(0)).float()
            indices = torch.argmax(condition, dim=1)
            selected_modulation = torch.stack([modulation_params[idx] for idx in indices])

        # Step 5: Self-attention and Modulation
        x = input + self.drop_path(self.self_attention(self.ln_1(input), modulation_params=selected_modulation))    #v1 model

        #x = input2 + input + self.drop_path(self.self_attention(self.ln_1(input), graph_features=scene_context1, modulation_params=selected_modulation))  #v2 model

        return x '''


class VSSBlock_graph(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            downscale: bool=True,
            condition_size: int = 4,
            use_modulation: bool = False,
            graph_input_dim: int =  26,   #v4 56, #v1-v3  28,  # Input dimension for graph node features
            graph_hidden_dim: int = 26, #v4 56, #v1-v3 28, # Hidden dimension for graph processing
            attr_dim: int = 3,          # Dimension of vehicle attributes (e.g., speed, color, etc.)
            use_attributes: bool = True, # use global attributes
            **kwargs,
    ):
        super().__init__()
        num_class = 5
        self.use_modulation = use_modulation
        self.ln_1 = norm_layer(hidden_dim)
        self.use_attributes = use_attributes
        
        self.graph_input_dim = graph_input_dim
        self.graph_proj = None
        # Validate hidden_dim and d_state
        if d_state <= 0 or hidden_dim <= 0:
            raise ValueError(f"Invalid d_state ({d_state}) or hidden_dim ({hidden_dim}). Both must be positive.")
        if hidden_dim % d_state != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by d_state ({d_state}).")
        

        # Self-attention layer for visual processing
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

        # Graph processing layers for vehicle skeletons
        self.conv1 = GCNConv(graph_input_dim, graph_hidden_dim)
        self.conv2 = GCNConv(graph_hidden_dim, graph_hidden_dim)
        
        #self.attr_fc = nn.Linear(attr_dim, graph_hidden_dim)

        self.graph_fc = nn.Linear(hidden_dim, hidden_dim)
        
        # Modulation network, if enabled
        if self.use_modulation:
            self.condition = nn.Parameter(torch.randn(condition_size, 128), requires_grad=True)
            self.mlp = nn.Sequential(
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 5)
            )

    #v3 model 
    
    def rescale_keypoints(self, keypoints, scale_factor):
        """
        Rescales keypoints according to the given scale factor.
        Args:
            keypoints (torch.Tensor): Tensor of shape (num_keypoints, 2) with (x, y) coordinates.
            scale_factor (float): Scaling factor (e.g., 0.5 for downsampling, 2.0 for upsampling).
        Returns:
            torch.Tensor: Rescaled keypoints.
        """
        return keypoints * scale_factor  # Works for both up and down scaling

    def scale_node_coordinates(self, data, W_feat, H_feat, W_orig=256, H_orig=256):
        scale = torch.tensor(
            [W_feat / W_orig, H_feat / H_orig], device=data.x.device
        )

        # Extract and rescale coordinates (last two features)
        coords = data.x[:, -2:] * scale

        # Clamp (avoid in-place ops)
        coords = torch.stack([
            coords[:, 0].clamp(0, W_feat - 1),
            coords[:, 1].clamp(0, H_feat - 1)
        ], dim=1)

        # Rebuild x (no in-place ops)
        new_x = torch.cat([data.x[:, :-2], coords], dim=1)

        # Clone to fully break graph connection (required for DDP safety)
        data.x = new_x.clone().detach().requires_grad_(data.x.requires_grad)

        return data


    '''def forward(self, input: torch.Tensor, condition: torch.Tensor, vehicles_graphs: Batch, attributes: torch.Tensor):
        """
        Forward pass of the extended VSSBlock with batched vehicle skeleton graphs and attributes.

        Parameters:
        - input: Visual feature tensor (e.g., from an image).
        - condition: Condition tensor for modulation.
        - vehicles_graphs: Batched vehicle graphs, where each vehicle is annotated with COCO-format skeleton (24 vertices).
        - attributes: Attribute tensor for each vehicle (e.g., speed, color), shape (num_vehicles, attr_dim).
        """

        if vehicles_graphs.x.shape[0] == 0:
            # ✅ Skip GCN processing if the graph is empty
            print("⚠️ No vehicle graphs available. Skipping GCN processing.")
            # Step 4: Conditional Modulation (if enabled)
            selected_modulation = None
            if self.use_modulation:
                modulation_params = self.mlp(self.condition)
                if condition.dim() == 1:
                    condition = F.one_hot(condition, num_classes=self.condition.size(0)).float()
                indices = torch.argmax(condition, dim=1)
                selected_modulation = torch.stack([modulation_params[idx] for idx in indices])

            # Step 5: Self-attention and Modulation
            x = input + self.drop_path(self.self_attention(self.ln_1(input), modulation_params=selected_modulation))
            return x
        else:
            if input.dim() == 4:
                # CNN-style input: [B, C, H, W]
                _, H_feat, W_feat, _ = input.shape

            vehicles_graphs = self.scale_node_coordinates(vehicles_graphs, W_feat=W_feat, H_feat=H_feat)

            x, edge_index, pos, batch = (
                vehicles_graphs.x, 
                vehicles_graphs.edge_index, 
                vehicles_graphs.pos, 
                vehicles_graphs.batch  # ✅ Extract `batch` for global_mean_pool
            )


        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        ##vehicle_embeddings = global_mean_pool(x, batch)  # Aggregate node features per graph/vehicle  OLD Implementation
        vehicle_embeddings = x  # [num_nodes, graph_hidden_dim]

        ## OLD implementation
        ##if self.use_attributes and attributes is not None:
        ##   num_graphs = batch.max().item() + 1  # Get the actual number of graphs
        ##    if attributes.shape[0] > num_graphs:
        ##        #print(f"⚠ Mismatched attribute size! Trimming from {attributes.shape[0]} to {num_graphs}")
        ##        attributes = attributes[:num_graphs]  # Trim extra attributes
        ##    attributes_emb = F.relu(self.attr_fc(attributes))  # Transform attributes to match graph feature dimension
        ##    vehicle_features = vehicle_embeddings + attributes_emb  # Combine node and attribute features
        ##else:
        ##    vehicle_features = vehicle_embeddings

        # === Bilinear Sampling with Full Batch ===
        B, H_feat, W_feat, C = input.shape
        input_perm = input.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Normalize node positions to [-1, 1]
        coords = vehicles_graphs.x[:, -2:].clone()
        coords[:, 0] = (coords[:, 0] / (W_feat - 1)) * 2 - 1
        coords[:, 1] = (coords[:, 1] / (H_feat - 1)) * 2 - 1

        batch_idx = vehicles_graphs.batch  # [num_nodes]
        # Expand batch_idx so we know which image each node belongs to
        num_nodes = coords.size(0)
        sampled_features = torch.zeros((num_nodes, C), device=input.device)

        input_perm = input.permute(0, 3, 1, 2)  # [B, C, H, W]

        for b in range(B):
            # Get all node indices belonging to image b
            node_mask = (batch_idx == b)
            if node_mask.sum() == 0:
                continue  # No nodes for this image

            coords_b = coords[node_mask]  # [num_nodes_in_b, 2]
            coords_b = coords_b.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]
            input_b = input_perm[b:b+1]  # [1, C, H, W]

            sampled_b = F.grid_sample(
                input_b,
                coords_b,
                mode='bilinear',
                align_corners=True
            )  # [1, C, N, 1]

            sampled_b = sampled_b.squeeze(3).squeeze(0).permute(1, 0)  # [N, C]

            sampled_features[node_mask] = sampled_b  # ⚡ Save sampled features aligned with node indices

        # === Fusion ===


        #vehicle_embeddings = global_mean_pool(x, batch)
        # Check if graph_proj exists, else initialize
        if self.graph_proj is None:
            graph_feature_dim = vehicle_embeddings.size(-1)  # e.g., 26
            sampled_feature_dim = sampled_features.size(-1)  # e.g., 96
            self.graph_proj = torch.nn.Linear(graph_feature_dim, sampled_feature_dim).to(vehicle_embeddings.device)

        # Project node features
        vehicle_embeddings_proj = self.graph_proj(vehicle_embeddings)  # [num_nodes, sampled_feature_dim]

        # Fuse
        fused_features = sampled_features + vehicle_embeddings_proj  # [num_nodes, sampled_feature_dim]

        ##scene_context = vehicle_features.mean(dim=0, keepdim=True)  # Overall context for all vehicles in scene OLD implemenetaion

        scene_context = global_mean_pool(fused_features, batch)  # [B, C]
        scene_context = self.graph_fc(scene_context)
        scene_context = scene_context.unsqueeze(-1).unsqueeze(-1) 

        input_perm = input_perm + scene_context  # Fuse
        input = input_perm.permute(0, 2, 3, 1)  # [B, H, W, C]


        # Project the scene context to match the visual input dimension
        ##scene_context = self.graph_fc(scene_context)   OLD Implementation
        ##input = input + scene_context  # Fuse scene context with visual features  OLD Implementation
        selected_modulation = None
        if self.use_modulation:
            modulation_params = self.mlp(self.condition)
            if condition.dim() == 1:
                condition = F.one_hot(condition, num_classes=self.condition.size(0)).float()
            indices = torch.argmax(condition, dim=1)
            selected_modulation = torch.stack([modulation_params[idx] for idx in indices])

        # Step 5: Self-attention and Modulation
        x = input + self.drop_path(self.self_attention(self.ln_1(input), modulation_params=selected_modulation))    #v1 model
        return x '''

    def forward(self, input: torch.Tensor, condition: torch.Tensor, vehicles_graphs: Batch, attributes: torch.Tensor):
        """
        Forward pass with bilinear feature sampling and multiple graphs per image handling.
        """

        if vehicles_graphs.x.shape[0] == 0:
            # No vehicle graphs fallback
            print("⚠️ No vehicle graphs available. Skipping GNN processing.")
            selected_modulation = None
            if self.use_modulation:
                modulation_params = self.mlp(self.condition)
                if condition.dim() == 1:
                    condition = F.one_hot(condition, num_classes=self.condition.size(0)).float()
                indices = torch.argmax(condition, dim=1)
                selected_modulation = torch.stack([modulation_params[idx] for idx in indices])

            x = input + self.drop_path(self.self_attention(self.ln_1(input), modulation_params=selected_modulation))
            return x

        else:
            if input.dim() == 4:
                B, H_feat, W_feat, C = input.shape  # [B, H, W, C]

            vehicles_graphs = self.scale_node_coordinates(vehicles_graphs, W_feat=W_feat, H_feat=H_feat)

            x, edge_index, pos, batch = (
                vehicles_graphs.x,
                vehicles_graphs.edge_index,
                vehicles_graphs.pos,
                vehicles_graphs.batch
            )

        # === Graph Neural Network Processing ===
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        vehicle_embeddings = x  # [num_nodes, graph_hidden_dim]

        # === Attribute Fusion (Optional) ===
        #if self.use_attributes and attributes is not None:
        #    num_graphs = batch.max().item() + 1
        #    if attributes.shape[0] > num_graphs:
        #        attributes = attributes[:num_graphs]
        #    attributes_emb = F.relu(self.attr_fc(attributes))
        #    graph_embeddings = global_mean_pool(vehicle_embeddings, batch)
        #    vehicle_features = graph_embeddings + attributes_emb
        #else:
        #    vehicle_features = global_mean_pool(vehicle_embeddings, batch)

        # === Bilinear Sampling ===
        input_perm = input.permute(0, 3, 1, 2)  # [B, C, H, W]

        coords = vehicles_graphs.x[:, -2:].clone()
        coords[:, 0] = (coords[:, 0] / (W_feat - 1)) * 2 - 1
        coords[:, 1] = (coords[:, 1] / (H_feat - 1)) * 2 - 1
        batch_idx = vehicles_graphs.batch

        num_nodes = coords.size(0)
        sampled_features = torch.zeros((num_nodes, C), device=input.device)

        for b in range(B):
            node_mask = (batch_idx == b)
            if node_mask.sum() == 0:
                continue
            coords_b = coords[node_mask].unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]
            input_b = input_perm[b:b+1]  # [1, C, H, W]

            sampled_b = F.grid_sample(
                input_b,
                coords_b,
                mode='bilinear',
                align_corners=True
            )  # [1, C, N, 1]

            sampled_b = sampled_b.squeeze(3).squeeze(0).permute(1, 0)  # [N, C]
            sampled_features[node_mask] = sampled_b

        # === Dynamic Graph Feature Projection ===
        if self.graph_proj is None:
            graph_feature_dim = vehicle_embeddings.size(-1)
            sampled_feature_dim = sampled_features.size(-1)
            self.graph_proj = torch.nn.Linear(graph_feature_dim, sampled_feature_dim).to(vehicle_embeddings.device)

        vehicle_embeddings_proj = self.graph_proj(vehicle_embeddings)  # [total_nodes, C]

        # === Fusion ===
        fused_features = sampled_features + vehicle_embeddings_proj  # [total_nodes, C]

        # === Scene Context per Image ===
        graph_features = global_mean_pool(fused_features, batch)  # [num_graphs, C]

        # Build mapping from graphs to images
        graph_to_image_batch_idx = vehicles_graphs.batch.new_zeros(graph_features.size(0))
        graph_counter = 0
        current_image_idx = vehicles_graphs.batch[0].item()

        for i in range(graph_features.size(0)):
            graph_to_image_batch_idx[i] = current_image_idx
            if i+1 < vehicles_graphs.batch.size(0):
                if vehicles_graphs.batch[i+1] != current_image_idx:
                    current_image_idx = vehicles_graphs.batch[i+1].item()

        # Average all graphs belonging to each image
        scene_context = torch.zeros((B, graph_features.size(-1)), device=graph_features.device)
        for b in range(B):
            mask = (graph_to_image_batch_idx == b)
            if mask.sum() > 0:
                scene_context[b] = graph_features[mask].mean(dim=0)

        scene_context = self.graph_fc(scene_context)  # [B, C]
        scene_context = scene_context.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        input_perm = input_perm + scene_context
        input = input_perm.permute(0, 2, 3, 1)  # [B, H, W, C]

        # === Modulation and Attention ===
        selected_modulation = None
        if self.use_modulation:
            modulation_params = self.mlp(self.condition)
            if condition.dim() == 1:
                condition = F.one_hot(condition, num_classes=self.condition.size(0)).float()
            indices = torch.argmax(condition, dim=1)
            selected_modulation = torch.stack([modulation_params[idx] for idx in indices])

        x = input + self.drop_path(self.self_attention(self.ln_1(input), modulation_params=selected_modulation))
        return x

    
    def bilinear_sample_features(self, input, pos, batch):
        """
        Bilinear interpolation to extract spatial features at graph node positions.
        
        - input: Visual feature tensor (B, C, H, W)
        - pos: Normalized node positions (N, 2), values in range [0,1]
        - batch: Batch index tensor mapping each node to its batch.

        Returns:
        - Sampled visual features (N, C)
        """
        B, C, H, W = input.shape  # Image feature tensor (B, C, H, W)
        N = pos.shape[0]



        # Convert normalized positions (0-1) to [-1, 1] for grid_sample
        pos = pos * 2 - 1  # Scale to [-1,1] for PyTorch grid_sample

        # Ensure batch alignment
        num_real_nodes = [(batch == b).sum().item() for b in range(B)]
        max_nodes = max(num_real_nodes)

        pos_batched = []
        for b in range(B):
            pos_b = pos[batch == b]  # Select positions for batch `b`

            # If this batch has fewer nodes, pad with (0,0)
            if pos_b.shape[0] < max_nodes:
                pad_size = max_nodes - pos_b.shape[0]
                pad = torch.zeros((pad_size, 2), device=pos.device)
                pos_b = torch.cat([pos_b, pad], dim=0)

            pos_b = pos_b.unsqueeze(0).unsqueeze(2)  # Shape: (1, max_nodes, 1, 2)
            pos_batched.append(pos_b)

        pos_batched = torch.cat(pos_batched, dim=0)  # Shape: (B, max_nodes, 1, 2)



        # ✅ Apply grid sampling
        sampled_features = F.grid_sample(input, pos_batched, mode="bilinear", align_corners=True)  # (B, C, 1, max_nodes)



        #print(f"Before squeezing: sampled_features.shape = {sampled_features.shape}")

        # 🔥 Ensure proper squeezing
        sampled_features = sampled_features.squeeze(3)  # Remove extra singleton dimension
        #print(f"After squeezing: sampled_features.shape = {sampled_features.shape}")

        # 🔥 Ensure the correct ordering of dimensions
        sampled_features = sampled_features.permute(0, 2, 1)  # (B, max_nodes, C)
        #print(f"After permute: sampled_features.shape = {sampled_features.shape}")

        # ✅ Remove padded nodes
        filtered_features = []
        for b in range(B):
            filtered_features.append(sampled_features[b, :num_real_nodes[b], :])  # Remove padded nodes

        sampled_features = torch.cat(filtered_features, dim=0)  # (N_real, C)

        #print(f"Final sampled_features.shape = {sampled_features.shape}")

        return sampled_features  # Ensure this has shape (N, C)




    '''
    def forward(self, input: torch.Tensor, condition: torch.Tensor, vehicles_graphs: Batch, attributes: torch.Tensor, cnn_features: torch.Tensor = None):
        """
        Forward pass of the extended VSSBlock with batched vehicle skeleton graphs and attributes.

        Parameters:
        - input: Visual feature tensor (e.g., from an image).
        - condition: Condition tensor for modulation.
        - vehicles_graphs: Batched vehicle graphs, where each vehicle is annotated with COCO-format skeleton (24 vertices).
        - attributes: Attribute tensor for each vehicle (e.g., speed, color), shape (num_vehicles, attr_dim).
        """
        
        # Step 1: Process vehicle skeleton graphs in batch
        # Extract Graph Features
        x, edge_index, pos, batch = (
            vehicles_graphs.x, 
            vehicles_graphs.edge_index, 
            vehicles_graphs.pos, 
            vehicles_graphs.batch  # ✅ Extract `batch` for global_mean_pool
        )

        #print(f"x={x.shape}")
        #print(f"pos={pos.shape}")
        #print(f"batch={batch.shape}")

        # ✅ Skip GCN processing if the graph is empty
        if vehicles_graphs.x.shape[0] == 0:
            #print("⚠️ No vehicle graphs available. Skipping GCN processing.")
            # Step 4: Conditional Modulation (if enabled)
            selected_modulation = None
            if self.use_modulation:
                modulation_params = self.mlp(self.condition)
                if condition.dim() == 1:
                    condition = F.one_hot(condition, num_classes=self.condition.size(0)).float()
                indices = torch.argmax(condition, dim=1)
                selected_modulation = torch.stack([modulation_params[idx] for idx in indices])

            # Step 5: Self-attention and Modulation
            x = input + self.drop_path(self.self_attention(self.ln_1(input), modulation_params=selected_modulation))
            return x
        #else:
        #    print("✅ Vehicle graphs available. Involving GCN processing.")

        if pos is not None:
            # ✅ Extract Image Features at Graph Node Positions

            

            sampled_features = self.bilinear_sample_features(input, pos, batch)  # Extract spatial features at graph node positions



            # ✅ Initialize projection layer dynamically
            if self.visual_feature_projection is None or not hasattr(self, 'visual_feature_projection'):
                if not hasattr(self, 'input_feature_dim') or self.input_feature_dim is None:
                    self.input_feature_dim = sampled_features.shape[1]  # Extract feature dimension dynamically
                self.visual_feature_projection = nn.Linear(self.input_feature_dim, 28).to(sampled_features.device)

            # 🚨 Debug batch indices
            #print(f"batch.shape={batch.shape}, batch.min()={batch.min()}, batch.max()={batch.max()}")
            #print(f"sampled_features.shape={sampled_features.shape}")


            sampled_features = self.visual_feature_projection(sampled_features)  # Align with graph_input_dim
            # 🔥 Concatenate Bilinear Sampled Image Features with Graph Nodes
            
            if sampled_features.shape[0] == batch.max().item() + 1:
                sampled_features = sampled_features[batch]  # ✅ Only use if batch is per-graph
            
            #print(f"Sampled features={sampled_features.shape[0]}")
            
            if sampled_features.shape[0] == 0:
                #print("⚠ Warning: sampled_features is empty! Creating a dummy tensor to avoid errors.")
                sampled_features = torch.zeros((x.shape[0], x.shape[1]), device=x.device)  # Dummy tensor

            expand_factor = x.shape[0] // sampled_features.shape[0]  # Get repeat factor
            
            remainder = x.shape[0] % sampled_features.shape[0]  # Get remainder

            sampled_features = sampled_features.repeat_interleave(expand_factor, dim=0)  # Expand
            if remainder > 0:
                sampled_features = torch.cat([sampled_features, sampled_features[:remainder]], dim=0)  # Pad remainder

            # ✅ Now, the batch size matches exactly
            assert sampled_features.shape[0] == x.shape[0], f"Mismatch! x.shape={x.shape[0]}, sampled_features.shape={sampled_features.shape[0]}"

            #print(f"✅ After expansion: sampled_features.shape = {sampled_features.shape}")

            #print(f"sampled_features.shape={sampled_features.shape}")
            #print(f"x.shape={x.shape}")
            #print("=========")
            # Now, concatenate safely
            x = torch.cat([x, sampled_features], dim=-1)  # ✅ Now safe to concatenate
        else:
            # v1-v3 model
            if pos is not None and pos.shape[0] == x.shape[0]:
                x = torch.cat([x, pos], dim=1)  # Append position data to node features




        if cnn_features is not None:
            # ✅ Apply RoI Align to extract CNN features at node positions
            B, C, H, W = cnn_features.shape
            N = pos.shape[0]

            # Convert normalized (0-1) positions to pixel space
            node_coords = pos * torch.tensor([W, H]).to(pos.device)
            node_coords = node_coords.view(N, 2)

            # Create RoI proposals (batch_index, x, y, width, height)
            rois = torch.cat([batch.view(-1, 1).float(), node_coords, torch.ones_like(node_coords)], dim=-1)  # (N, 5)
            cnn_sampled = self.roi_align(cnn_features, rois)  # (N, C, 1, 1)
            cnn_sampled = cnn_sampled.view(N, C)  

            # 🔥 Project CNN features to match graph input dimensions
            cnn_sampled = self.cnn_feature_projection(cnn_sampled)

            # Concatenate CNN features with graph node features
            x = torch.cat([x, cnn_sampled], dim=-1)  # 🔥 Spatially enhanced features

        # Apply GCN layers to each vehicle skeleton
        # v1 and v2 models
        #x = F.relu(self.conv1(x, edge_index))
        #x = F.relu(self.conv2(x, edge_index))

        if x.shape[1] != self.graph_input_dim:
            projection_layer = nn.Linear(x.shape[1], self.graph_input_dim).to(x.device)  # Match expected size
            x = projection_layer(x)

        # v3 of a model with using graph attention nn
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # v1 and v2 models
        # Aggregate features across all vertices for each vehicle
        vehicle_embeddings = global_mean_pool(x, batch)  # Aggregate node features per graph/vehicle

        #v3 model
        #vehicle_embeddings = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch), global_add_pool(x, batch)], dim=-1)



        #print(f"batch.max() + 1 = {batch.max().item() + 1}")
        #print(f"attributes.shape[0] = {attributes.shape[0]}")

        if self.use_attributes and attributes is not None:
            # Step 3: Ensure attributes match the number of graphs (trim if necessary)
            

            #num_graphs = torch.unique(batch).numel()
            num_graphs = batch.max().item() + 1  # Get the actual number of graphs
            
            #print(f"Before Trimming: Attributes Shape: {attributes.shape}, Num Graphs (Computed): {num_graphs}")
            #print(f"Batch Tensor: {batch}, Shape: {batch.shape}, Max: {batch.max().item()}")
            
            if attributes.shape[0] > num_graphs:
                #print(f"⚠ Mismatched attribute size! Trimming from {attributes.shape[0]} to {num_graphs}")
                attributes = attributes[:num_graphs]  # Trim extra attributes

            # Step 2: Process attributes and combine with graph features
            attributes_emb = F.relu(self.attr_fc(attributes))  # Transform attributes to match graph feature dimension
            # v1 and v2 models
            vehicle_features = vehicle_embeddings + attributes_emb  # Combine node and attribute features
            #vehicle_features = torch.cat([vehicle_embeddings, attributes_emb], dim=-1)  # Shape: (batch_size, graph_hidden_dim + attr_dim)
            #vehicle_features = self.fusion_layer(vehicle_features)  # Apply a learnable transformation
        else:
            vehicle_features = vehicle_embeddings

        # Step 3: Pool vehicle-level embeddings to get a scene-level context
        scene_context1 = vehicle_features.mean(dim=0, keepdim=True)  # Overall context for all vehicles in scene

        # Project the scene context to match the visual input dimension
        scene_context2 = self.graph_fc(scene_context1)
        
        #input = input + scene_context  # Fuse scene context with visual features  <-- is this one wrong?    v1 model
        

        input2 = input + scene_context2  #v2 model
        #print(f"scene_content1.shape={scene_context1.shape}")
        #print(f"scene_content2.shape={scene_context2.shape}")

        #fusion_input = torch.cat([input, scene_context], dim=-1)  # Concatenate features
        #gate = torch.sigmoid(self.fusion_gate(fusion_input))  # Compute gate values (between 0 and 1)
        #input2 = gate * input + (1 - gate) * scene_context  # Weighted fusion


        # the result should be called input2

        # Step 4: Conditional Modulation (if enabled)
        selected_modulation = None
        if self.use_modulation:
            modulation_params = self.mlp(self.condition)
            if condition.dim() == 1:
                condition = F.one_hot(condition, num_classes=self.condition.size(0)).float()
            indices = torch.argmax(condition, dim=1)
            selected_modulation = torch.stack([modulation_params[idx] for idx in indices])

        # Step 5: Self-attention and Modulation
        #x = input + self.drop_path(self.self_attention(self.ln_1(input), modulation_params=selected_modulation))    v1 model

        # v3 model
        #x = input2 + input + self.drop_path(self.self_attention(self.ln_1(input), graph_features=scene_context1, modulation_params=selected_modulation))  #v2 model

        x = input2 + self.drop_path(self.self_attention(self.ln_1(input2), graph_features=None, modulation_params=selected_modulation))  #v2 model

        return x'''


class VSSLayer_graph(nn.Module):
    """ A basic layer incorporating multiple VSSBlocks for one stage.
    
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        d_state (int): State size for attention mechanism in each block.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # Initialize a list of VSSBlocks within this layer
        self.blocks = nn.ModuleList([
            VSSBlock_graph(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                downscale=True,
                **kwargs  # Pass any additional arguments to VSSBlock
            )
            for i in range(depth)
        ])

        # Initialize weights
        def _init_weights(module: nn.Module):
            for name, p in module.named_parameters():
                if name == "out_proj.weight":
                    p = p.clone().detach_()
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))

        self.apply(_init_weights)

        # Optional downsample layer
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x, condition, vehicles_graphs=None, attributes=None):
        """
        Forward pass through the VSSLayer.
        
        Args:
            x (torch.Tensor): Input visual features.
            condition (torch.Tensor): Condition vector for modulation in each VSSBlock.
            vehicles_graphs (torch_geometric.data.Batch, optional): Batched vehicle graphs with skeleton information.
            attributes (torch.Tensor, optional): Attributes matrix for each vehicle (e.g., speed, color).

        Returns:
            torch.Tensor: Processed output after applying VSSBlocks and optional downsampling.
        """
        
        # Pass through each VSSBlock in the layer
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, condition, vehicles_graphs, attributes)

        # Apply optional downsample layer
        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSLayer_up_graph(nn.Module):
    """ A basic layer incorporating multiple VSSBlocks with an upsampling option for one stage.
    
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): Upsample layer at the start of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        d_state (int): State size for attention mechanism in each block.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # Initialize a list of VSSBlocks with modulation enabled
        self.blocks = nn.ModuleList([
            VSSBlock_graph(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                use_modulation=True,  # Enable modulation for upsampling stage
                downscale=False,
                **kwargs  # Pass additional arguments to VSSBlock
            )
            for i in range(depth)
        ])

        # Optional upsample layer
        self.upsample = upsample(dim=dim, norm_layer=norm_layer) if upsample is not None else None

    def forward(self, x, condition, vehicles_graphs=None, attributes=None):
        """
        Forward pass through the VSSLayer_up.
        
        Args:
            x (torch.Tensor): Input visual features.
            condition (torch.Tensor): Condition vector for modulation in each VSSBlock.
            vehicles_graphs (torch_geometric.data.Batch, optional): Batched vehicle graphs with skeleton information.
            attributes (torch.Tensor, optional): Attributes matrix for each vehicle (e.g., speed, color).

        Returns:
            torch.Tensor: Processed output after applying upsampling (if defined) and VSSBlocks.
        """
        
        # Apply upsampling layer, if defined
        if self.upsample is not None:
            x = self.upsample(x)

        # Pass through each VSSBlock in the layer
        for blk in self.blocks:
            if self.use_checkpoint:
                # Use checkpointing for memory efficiency
                x = checkpoint.checkpoint(lambda *inputs: blk(*inputs), x, condition, vehicles_graphs, attributes)
            else:
                # Call VSSBlock with additional graph and attribute inputs
                x = blk(x, condition, vehicles_graphs, attributes)

        return x


class VSSM_graph(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)

        # Absolute Position Embedding (ape) setup
        self.ape = False
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        # Encoder layers
        self.layers = nn.ModuleList()
        #print("self.num_layers=", self.num_layers)
        for i_layer in range(self.num_layers):
            #print("i_layer=", dims[i_layer])
            #print("i_layer=", depths[i_layer])
            layer = VSSLayer_graph(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # Decoder layers with VSSLayer_up
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up_graph(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, num_classes, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, condition, vehicles_graphs=None, attributes=None):
        """
        Forward pass through encoder layers with graph and attribute data support.
        """
        skip_list = []
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # Pass through encoder layers
        for layer in self.layers:
            skip_list.append(x)
            if vehicles_graphs is not None and attributes is not None:
                x = layer(x, condition, vehicles_graphs, attributes)  # Include graph and attributes
            else:
                x = layer(x, condition)  # Fallback without additional inputs
        return x, skip_list

    def forward_features_up(self, x, skip_list, condition, vehicles_graphs=None, attributes=None):
        """
        Forward pass through decoder layers, including graph and attribute data.
        """
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x, condition, vehicles_graphs, attributes)
            else:
                x = layer_up(x + skip_list[-inx], condition, vehicles_graphs, attributes)
        return x

    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        x = self.final_conv(x)
        return x

    def forward(self, x, condition, vehicles_graphs=None, attributes=None):
        """
        Forward pass through VSSM.

        Args:
            x (torch.Tensor): Input image features.
            condition (torch.Tensor): Condition vector for modulation.
            vehicles_graphs (torch_geometric.data.Batch, optional): Batched vehicle graphs with skeleton information.
            attributes (torch.Tensor, optional): Attributes matrix for each vehicle (e.g., speed, color).

        Returns:
            torch.Tensor: Final output after processing with VSSM.
        """
        # Encoder path
        x, skip_list = self.forward_features(x, condition, vehicles_graphs, attributes)

        # Decoder path with graph and attribute data
        x = self.forward_features_up(x, skip_list, condition, vehicles_graphs, attributes)

        # Final output
        x = self.forward_final(x)

        return x

