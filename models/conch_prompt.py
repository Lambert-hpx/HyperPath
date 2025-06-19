import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.topk import PerturbedTopK, HardtopK
from fast_pytorch_kmeans import KMeans
import random
import numpy as np
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
from fast_pytorch_kmeans import KMeans

from models.prompts import get_prompts
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

class UniformSampler(object):
    """Random sampling the points, return the indices for each unique subset, or in batch."""
    def __init__(self, batch_size, num_samples, ):
        self.batch_size = batch_size
        self.num_samples = num_samples
        # self.num_points = num_points

    def unique_generate(self, num_points):
        sample_indices = torch.tensor([random.randint(0, len(num_points) - 1) for _ in range(self.num_samples)])
        return sample_indices

    def batch_generate(self,num_points):
        sample_indices = torch.randint(0,
            num_points - 1,
            (self.batch_size, self.num_samples))
        return sample_indices

    def sample(self):
        sample_indices = self.batch_generate()
        return sample_indices

class GumbelSoftmaxSampler():
    """Sample based on a Gumbel-Max distribution.

    Use re-param trick for back-prop
    """
    def __init__(self, batch_size, num_samples, tau=0.1, device='cuda', data_type=torch.float32):
        self.batch_size = batch_size
        self.num_samples = num_samples
        # self.num_points = num_points
        self.device = device
        self.dtype = data_type
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(
                torch.tensor(0., device=self.device, dtype=self.dtype),
                torch.tensor(1., device=self.device, dtype=self.dtype))
        self.tau = tau

    def sample(self, logits=None, num_points=2000, selected=None):

        if logits==None:
            logits = torch.ones([self.batch_size, num_points], device=self.device, dtype=self.dtype, requires_grad=True)
        else:
            logits = logits.to(self.dtype).to(self.device).repeat([self.batch_size, 1])

        if selected is None:
            # import ipdb; ipdb.set_trace()
            logits = torch.log(logits + 1e-20)
            # print(logits.max() - logits.min())
            gumbels = self.gumbel_dist.sample(logits.shape)
            gumbels = (logits + gumbels)/self.tau
            y_soft = gumbels.softmax(-1)
            topk = torch.topk(gumbels, self.num_samples, dim=-1)
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, topk.indices, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
            # chosen_idxs = np.argpartition(-gumbels, self.num_samples)[:self.num_samples]
            # global_mask = np.zeros(len(gumbels)).astype(bool)
            # global_mask[chosen_idxs] = True
        else:
            pass

        return ret, y_hard, y_soft#, topk.indices

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

    flops = 0 # below code flops = 0
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
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

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
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # self.conv2d = nn.Conv2d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     groups=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=d_conv,
        #     padding=(d_conv - 1) // 2,
        #     **factory_kwargs,
        # )
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        # self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        # self.x_proj_weight = nn.Parameter(self.x_proj.weight) # (N, inner)
        # del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        # self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        # self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        # self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        # del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
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
        # import ipdb; ipdb.set_trace()
        B, C, L = x.shape
        # import ipdb; ipdb.set_trace()
        # L = H * W
        B = B * 2
        K = 1

        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
        # xs = x
        # import ipdb; ipdb.set_trace()
        xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=0) # (2*b, k//2, d, l)
        # xs = x
        # merge first batch dim
        # import ipdb; ipdb.set_trace()
        # xs = rearrange(xs, "b k d l -> 1 (b k) d l") # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # import ipdb; ipdb.set_trace()
        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        # import ipdb; ipdb.set_trace()
        assert out_y.dtype == torch.float

        y1 = out_y[0].view(B//2, -1, L)
        y2 = torch.flip(out_y[1], dims=[-1]).view(B//2, -1, L)
        # y3 = out_y[:, 2]
        # y4 = torch.flip(out_y[:, 3], dims=[-1]).view(B, -1, L)
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        # return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
        return y1, y2, dts

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

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

    def forward(self, x: torch.Tensor, **kwargs):
        # import ipdb; ipdb.set_trace()
        # B, H, W, C = x.shape
        B, L, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, l, d)

        # x = x.permute(0, 3, 1, 2).contiguous()
        # x = self.act(self.conv2d(x)) # (b, d, h, w)
        x = self.act(self.conv1d(x.transpose(1, 2).contiguous())) # (b, d, l)
        # y1, y2, y3, y4 = self.forward_core(x)
        y1, y2, dts = self.forward_core(x)
        # assert y1.dtype == torch.float32
        # y = y1 + y2 + y3 + y4
        # y = y1
        # y = torch.stack([y1+y2, y3+y4], dim=1).view(B, -1, L)
        # import ipdb; ipdb.set_trace()
        y = (y1 + y2) / 2
        # y = y1
        # import ipdb; ipdb.set_trace()
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = y.transpose(1, 2).contiguous()
        y = self.out_norm(y)
        y = y * F.silu(z)
        # y2 = y2.transpose(1, 2).contiguous()
        # y2 = self.out_norm(y2)
        # y2 = y2 * F.silu(z)
        # y = y1 + y2
        # y = torch.cat([y1, y2], dim=-1)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out, dts


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        out, dts = self.self_attention(self.ln_1(x))
        x = x + self.drop_path(out)
        return x, dts
        # x1, x2 = x
        # # x1 = rearrange(x1, "b h w c -> b (h w) c")
        # # x2 = rearrange(x2, "b c -> b 1 c")
        # x_dfs = torch.cat([rearrange(x2, "b c -> b 1 c"), rearrange(x1, "b h w c -> b (h w) c")], dim=1)
        # x_dfs = rearrange(x_dfs, "b l c -> 1 (b l) c")
        # x_bfs = torch.cat([rearrange(x2, "b c -> 1 b c"), rearrange(x1, "b h w c -> 1 (b h w) c")], dim=1)

        # x_all = torch.cat([x_dfs, x_bfs], dim=0)

        # # x1 = torch.cat([x2, x1], dim=1)
        # # import ipdb; ipdb.set_trace()
        # x = x_all + self.drop_path(self.self_attention(self.ln_1(x_all)))
        # x = rearrange(x[0] + x[1], "b c -> 1 b c")
        # return [x, x2]


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
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

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.topk = 4000
        # self.n_clusters = 8
        # self.kmeans = KMeans(n_clusters=self.n_clusters, mode='euclidean', verbose=1)
        # self.patch_sorter = PerturbedTopK(self.topk)
        self.attn_pool = Attn_Net_Gated(L=dim, D=dim, dropout=False, n_classes=1)
        self.pre_head = nn.Linear(dim, 2) if 2 > 0 else nn.Identity()
        self.sampler = GumbelSoftmaxSampler(1, self.topk)
        # self.score_net = [
        #     nn.Linear(dim, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 1)]
        # self.score_net = nn.Sequential(*self.score_net)

    def forward(self, x):
        # x1, x2 = x

        # x_dfs = torch.cat([rearrange(x2, "b c -> b 1 c"), rearrange(x1, "b h w c -> b (h w) c")], dim=1)
        # x_dfs = rearrange(x_dfs, "b l c -> 1 (b l) c")
        # x_bfs = torch.cat([rearrange(x2, "b c -> 1 b c"), rearrange(x1, "b h w c -> 1 (b h w) c")], dim=1)

        # x = torch.cat([x_dfs, x_bfs], dim=0)
        # x = rearrange(x1, "b h w c -> 1 (b h w) c")
        # x = x_bfs
        # x = x_dfs
        # x = rearrange(x1, "b h w c -> 1 (b h w) c")
        # print(x.shape)
        # x = x[:, :110000, :]
        # rand_idx = torch.randperm(x.shape[1])
        # x = x[:, torch.sort(rand_idx[:min(int(x.shape[1] * 0.4), 130000)])[0], :]
        # x = x[:, torch.sort(rand_idx[:10000])[0], :]

        # labels = self.kmeans.fit_predict(x[0])
        # import ipdb; ipdb.set_trace()
        # score, _ = self.attn_pool(x.squeeze(0))  # L 1
        # # score = self.score_net(x.squeeze(0))
        # score = torch.transpose(score, 1, 0)
        # score = F.softmax(score, dim=1)
        # x_pre = torch.flatten(torch.mm(score, x.squeeze(0)), 1)
        # x_pre = self.pre_head(x_pre)
        # # import ipdb; ipdb.set_trace()
        # if score.shape[1] > self.topk:
        #     # if self.training:
        #         # idx = self.patch_sorter(score)
        #     # if self.training:
        #     idx, idx_hard, idx_soft = self.sampler.sample(score)
        #     x = x[:, (idx != 0).squeeze(), :]
            # else:
            #     topk = torch.topk(score, self.topk, dim=-1)
            #     idx = torch.zeros_like(score, memory_format=torch.legacy_contiguous_format).scatter_(-1, topk.indices, 1.0)
            #     x = x[:, (idx != 0).squeeze(), :]
                # idx = HardtopK(score, self.topk)
                # import ipdb; ipdb.set_trace()
                # x = (x.transpose(1,2) @ idx).transpose(1,2)
                # x = x[:, (idx_hard != 0).squeeze(), :]
            # else:
                # idx = HardtopK(score, self.topk)
            # x = (x.transpose(1,2) @ idx).transpose(1,2)


        for blk in self.blocks:
            if self.use_checkpoint:
                out = checkpoint.checkpoint(blk, x)
            else:
                out, dts = blk(x)
        # import ipdb; ipdb.set_trace()
        
        if self.downsample is not None:
            out = self.downsample(out)

        return out, dts

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class VSSM(nn.Module):
    def __init__(self, n_classes=2, in_dim=16, depths=[1], 
                 dims=[256], d_state=16, drop_rate=0.2, attn_drop_rate=0., drop_path_rate=0.6,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        # self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        # self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            # norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.fc1 = nn.Sequential(nn.Linear(in_dim, dims[0]), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_dim, dims[0]), nn.ReLU())

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                # downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        # self.rho = nn.Sequential(*[nn.Linear(self.num_features, self.num_features), nn.ReLU(), nn.Dropout(0.25)])
        self.head = nn.Linear(self.num_features, n_classes) if n_classes > 0 else nn.Identity()
        # self.head_pre = nn.Linear(self.num_features, n_classes) if n_classes > 0 else nn.Identity()
        self.attn_pool = Attn_Net_Gated(L=self.num_features, D=self.num_features, dropout=False, n_classes=1)

        self.topk = 2000
        self.pre_attn = nn.Sequential(*[nn.Linear(in_dim, 512), nn.ReLU(), Attn_Net_Gated(L=512, D=256, dropout=False, n_classes=1)])
        # self.pre_attn_pool = Attn_Net_Gated(L=in_dim, D=in_dim, dropout=False, n_classes=1)
        self.pre_head = nn.Linear(in_dim, n_classes) if n_classes > 0 else nn.Identity()
        self.sampler = GumbelSoftmaxSampler(1, self.topk)

        # self.feature_mean = nn.Parameter(torch.zeros(in_dim).to("cuda"))
        self.register_buffer('feature_mean', torch.zeros(in_dim).to("cuda"))
        self.mean_num = 0

        self.gumbel_dist = torch.distributions.gumbel.Gumbel(
                torch.tensor(0., device="cuda"),
                torch.tensor(1., device="cuda"))

        self.apply(self._init_weights)

        self.conch, _ = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path="/home/yyhuang/WSI/HIT/models/weights/conch.bin")
        self.conch.eval()

        # tokenizer = get_tokenizer()
        # cls_templates = get_prompts()
        # self.text_features = []
        # for template in cls_templates:
        #     tokenized_templates = tokenize(texts=template, tokenizer=tokenizer)
        #     text_feature = self.conch.encode_text(tokenized_templates).detach().to("cuda")
        #     self.text_features.append(text_feature.mean(dim=0))
        # self.text_features = torch.stack(self.text_features, dim=0)

        tokenizer = get_tokenizer()
        cls_templates = get_prompts()
        tokenized_templates = tokenize(texts=cls_templates, tokenizer=tokenizer)
        self.text_features = self.conch.encode_text(tokenized_templates).detach().to("cuda")

        self.cls1_feat = self.text_features[:len(cls_templates)//2].mean(dim=0)
        self.cls2_feat = self.text_features[len(cls_templates)//2:].mean(dim=0)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def forward_features(self, x):
        # x = self.patch_embed(x)
        # if self.ape:
        #     x = x + self.absolute_pos_embed
        # x = self.pos_drop(x)
        # x1, x2 = x
        x = self.pos_drop(x)
        # x2 = self.pos_drop(x2)
        # x = [x1, x2]

        for layer in self.layers:
            # import ipdb; ipdb.set_trace()
            x, dts = layer(x)

        # x = torch.flatten(x, 1, 2) # B H W C -> B L C
        # x = rearrange(x, "b l c -> b (h w) c")
        # x = torch.cat([x[:1], x[1:]], dim=-1)
        # import ipdb; ipdb.set_trace()
            
        # x = x[:, -1, :] # get the last state
        # x = x[0]
            
        x = self.norm(x)  # B L C
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        x_max = self.maxpool(x.transpose(1, 2))  # B C 1
        # import ipdb; ipdb.set_trace()

        # A, h = self.attn_pool(x.squeeze(0))  # L 1
        # A = torch.transpose(A, 1, 0)
        # A = F.softmax(A, dim=1) 
        # x = torch.mm(A, x.squeeze(0))

        # A = rearrange(A, "1 (b h w) -> b h w", h=8, w=8)

        # x = torch.flatten(x, 1)
        x_max = torch.flatten(x_max, 1)
        return x_max, dts

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x1, x2):
        # import ipdb; ipdb.set_trace()
        # x1 = rearrange(x1, "b h w c -> b (h w) c")
        # x1 = rearrange(x1, "b l c -> 1 (b l) c")
        # x2 = rearrange(x2, "b c -> 1 b c")

        x1 = rearrange(x1, "b h w c -> 1 (b h w) c")
        # x_sim = (x1 @ self.text_features.T * self.conch.logit_scale.exp()).detach()
        # x = torch.cat([x1, x_sim], dim=-1)
        x = x_sim
        # x_sim = (x_sim - x_sim.mean(dim=2, keepdim=True)) / (x_sim.std(dim=2, keepdim=True) + 1e-8)
        # x_sim = F.normalize(x_sim, p=2, dim=(1, 2))
        # x_sim = (x_sim - x_sim.min())
        # x = torch.cat([x1])

        # x_dfs = torch.cat([rearrange(x2, "b c -> b 1 c"), rearrange(x1, "b h w c -> b (h w) c")], dim=1)
        # x_dfs = rearrange(x_dfs, "b l c -> 1 (b l) c")
        # x = x_dfs

        # x_bfs = torch.cat([rearrange(x2, "b c -> 1 b c"), rearrange(x1, "b h w c -> 1 (b h w) c")], dim=1)
        # x = x_bfs

        # self.topk = 2000
        # # self.topk = int(0.7 * x.shape[1])
        # if x.shape[1] > self.topk:
        #     if self.training:
        #         rand_idx = torch.randperm(x.shape[1])[:self.topk]
        #         x = x[:, torch.sort(rand_idx)[0], :]
        #         self.feature_mean = (self.feature_mean * self.mean_num + x.mean(dim=1).squeeze()) / (self.mean_num + 1)
        #         self.mean_num += 1

        #         # x = self.fc1(x)
        #         # x, dts = self.forward_features(x)
        #     else:
        #         self.wsi_feat = x.mean(dim=1).squeeze()
        #         q_feat = torch.matmul(F.normalize(self.wsi_feat.unsqueeze(0).unsqueeze(0), p=2, dim=2), F.normalize(x.transpose(1, 2).contiguous(), p=2, dim=1))
        #         p_feat = torch.matmul(F.normalize(self.feature_mean.unsqueeze(0).unsqueeze(0), p=2, dim=2), F.normalize(x.transpose(1, 2).contiguous(), p=2, dim=1))
        #         ratios = (p_feat / (q_feat + 1e-8)).squeeze()
        #         weights = (ratios - ratios.min()) / (ratios.max() - ratios.min())
        #         # weights = torch.softmax(ratios, dim=0)
        #         log_weights = torch.log(weights + 1e-8)
        #         # log_ratio = (torch.log(p_feat + 1e-8) - torch.log(q_feat + 1e-8)).squeeze()
        #         # weights = torch.softmax(log_ratio, dim=0)
        #         # gumbel = self.gumbel_dist.sample(log_ratio.shape)
        #         gumbel = self.gumbel_dist.sample(weights.shape)
        #         log_weights = log_weights + gumbel
        #         topk_idx = torch.topk(log_weights, self.topk, dim=-1).indices
        #         x = x[:, torch.sort(topk_idx)[0], :]

        x = self.fc1(x)
        x, dts = self.forward_features(x)
            # self.log_diff = torch.log(self.feature_mean + 1e-8) - torch.log(self.wsi_feat + 1e-8)
            # import ipdb; ipdb.set_trace()
            # self.importance = torch.mul(self.log_diff.unsqueeze(0).unsqueeze(-1), x.transpose(1, 2).contiguous())
            # import ipdb; ipdb.set_trace()
        #     # x = self.rho(x)
        #     x = self.head(x)
        #     # x_patch = self.head(A)
        #     # dts = rearrange(dts[0], "c (b h w) -> c b h w", h=8, w=8)
        #     dts = dts[0]
        # else:
        # x_all = []
        # rand_idx_all = torch.randperm(x.shape[1])
        # rand_idx_all = torch.concat([rand_idx_all, rand_idx_all, rand_idx_all], dim=0)
        # bag_num= 1
        # bag_size = 4000
        # train_num = bag_num * bag_size
        # times = train_num // rand_idx_all.shape[0] + 1
        # import ipdb; ipdb.set_trace()
        # rand_idx_all = torch.concat([rand_idx_all]*times, dim=0)
        # for i in range(bag_num):
        #     rand_idx = rand_idx_all[i*bag_size:(i+1)*bag_size]
        #     # import ipdb; ipdb.set_trace()
        #     x_t = x[:, torch.sort(rand_idx)[0], :]
        #     x_t = self.fc1(x_t)
        #     x_t, dts = self.forward_features(x_t)
        #     x_all.append(x_t)
        # import ipdb; ipdb.set_trace()
        # x = torch.cat(x_all, dim=0)
        # A, _ = self.pre_attn(x.squeeze(0))  # L 1
        # A = torch.transpose(A, 1, 0)
        # A = F.softmax(A, dim=1) 
        # x_pre = torch.mm(A, x.squeeze(0))
        # x_pre = self.pre_head(x_pre)
        
        # attn_score_sort, sort_idx = torch.sort(A.squeeze())
        # attn_score_cumsum = torch.cumsum(attn_score_sort, dim=0)
        # bag_size = 2000
        # score_idx = torch.linspace(0, 1, bag_size+1, device=attn_score_cumsum.device)[:-1]
        # diff = torch.abs(attn_score_cumsum.unsqueeze(1) - score_idx)
        # closest_idx = torch.argmin(diff, dim=0)
        # patch_idx = torch.sort(sort_idx[closest_idx])[0]
        # # print(patch_idx.unique().shape[0])
        # x = x[:, patch_idx, :]
        # x = self.fc1(x)
        # x, dts = self.forward_features(x)
        # score_idx = (torch.linspace(0, 1, bag_size+1, device=attn_score_cumsum.device)[:-1]*x.shape[1]).type(torch.int)
        # patch_idx = sort_idx[score_idx]

        # import ipdb; ipdb.set_trace()
        # x = torch.max(x, dim=0)[0].unsqueeze(0)
        # rand_idx1 = torch.randperm(x.shape[1])[:4000]
        # rand_idx2 = torch.randperm(x.shape[1])[:4000]
        # rand_idx3 = torch.randperm(x.shape[1])[:4000]
        # rand_idx4 = torch.randperm(x.shape[1])[:4000]
        # x1 = x[:, torch.sort(rand_idx1)[0], :]
        # x2 = x[:, torch.sort(rand_idx2)[0], :]
        # x3 = x[:, torch.sort(rand_idx3)[0], :]
        # x4 = x[:, torch.sort(rand_idx4)[0], :]

        # x1 = self.fc1(x1)
        # x2 = self.fc1(x2)
        # x3 = self.fc1(x3)
        # x4 = self.fc1(x4)

        # x1, dts = self.forward_features(x1)
        # x2, _ = self.forward_features(x2)
        # x3, _ = self.forward_features(x3)
        # x4, _ = self.forward_features(x4)

        # x = torch.max(torch.concat([x1, x2], dim=0), dim=0)[0].unsqueeze(0)
        # x = torch.max(torch.concat([x1, x2, x3], dim=0), dim=0)[0].unsqueeze(0)
        # x = torch.max(torch.concat([x1, x2, x3, x4], dim=0), dim=0)[0].unsqueeze(0)

        # x = torch.cat([x1, x2, x3], dim=1)
        x = self.head(x)
            # x = (x1 + x2 + x3) / 3
            # x = torch.max(torch.concat([x1, x2, x3], dim=0), dim=0)[0].unsqueeze(0)
        # import ipdb; ipdb.set_trace()
            # x = self.rho(x)
            # x1 = self.head(x1)
            # x2 = self.head(x2)
            # x3 = self.head(x3)
            # x1_max = torch.max(F.softmax(x1), dim=1)[0]
            # x2_max = torch.max(F.softmax(x2), dim=1)[0]
            # x3_max = torch.max(F.softmax(x3), dim=1)[0]
            # import ipdb; ipdb.set_trace()
            # select x with the highest score
            # if x1_max > x2_max:
            #     x = x1
            # else:
            #     x = x2
            # if x3_max > max(x1_max, x2_max):
            #     x = x3
            # x = x1

        # dts_region = dts[:, torch.tensor([i for i in torch.arange(dts.shape[-1]) if i % 65 == 0])]
        # dts_patch = dts[:, torch.tensor([i for i in torch.arange(dts.shape[-1]) if i % 65 != 0])]
        # import ipdb;ipdb.set_trace()
        # dts_patch = rearrange(dts_patch, "c (b h w) -> c b h w", h=8, w=8)
        # dts_patch = rearrange(dts_patch, "c (b h w) -> c b h w", h=16, w=16)

        # dts_region = dts[:, :x1.shape[0]]
        # dts_patch = dts[:, x1.shape[0]:]
        # dts_patch = rearrange(dts_patch, "c (b h w) -> c b h w", h=8, w=8)

        # import ipdb; ipdb.set_trace()
        # x_patch1 = rearrange(x_patch[:, :, 0], "1 (b h w) -> b h w", h=8, w=8)
        # x_patch2 = rearrange(x_patch[:, :, 1], "1 (b h w) -> b h w", h=8, w=8)
        Y_hat = torch.argmax(x)
        Y_prob = F.softmax(x)
        # import ipdb; ipdb.set_trace()
        return x, Y_hat, Y_prob, dts
        # return x, Y_hat, Y_prob, weights
        # return x, Y_hat, Y_prob, x_pre
        # return x, Y_hat, Y_prob, [dts_region, dts_patch]


# APIs with VMamba2Dp =================
def check_vssm_equals_vmambadp():
    from bak.vmamba_bak1 import VMamba2Dp

    # test 1 True =================================
    torch.manual_seed(time.time()); torch.cuda.manual_seed(time.time())
    oldvss = VMamba2Dp(depths=[2,2,6,2]).half().cuda()
    newvss = VSSM(depths=[2,2,6,2]).half().cuda()
    newvss.load_state_dict(oldvss.state_dict())
    input = torch.randn((12, 3, 224, 224)).half().cuda()
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y1 = oldvss.forward_backbone(input)
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y2 = newvss.forward_backbone(input)
    print((y1 -y2).abs().sum()) # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
    
    # test 2 True ==========================================
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    oldvss = VMamba2Dp(depths=[2,2,6,2]).cuda()
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    newvss = VSSM(depths=[2,2,6,2]).cuda()

    miss_align = 0
    for k, v in oldvss.state_dict().items(): 
        same = (oldvss.state_dict()[k] == newvss.state_dict()[k]).all()
        if not same:
            print(k, same)
            miss_align += 1
    print("init miss align", miss_align) # init miss align 0


if __name__ == "__main__":
    check_vssm_equals_vmambadp()