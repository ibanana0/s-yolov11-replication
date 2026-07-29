# Modul GhostNet (v1 & v3) untuk S-YOLOv11.
# File ini menyediakan C3k2_Ghost (GhostNet Base) dan C3k2_GhostV3 (GhostNet-v3).
# Salin ke ultralytics/nn/modules_ghost.py dan daftarkan melalui apply_patch.py.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


# ==============================================================================
# 1. GHOSTNET BASE (v1) MODULES
# ==============================================================================

class GhostConv(nn.Module):
    """Ghost Convolution v1 (Han et al., CVPR 2020)."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2  # intrinsic channels
        self.cv1 = Conv(c1, c_, k, s, p=None, g=g, act=act)
        self.cv2 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)  # cheap depthwise conv

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), dim=1)


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck v1."""

    def __init__(self, c1, c2, k=3, s=1, shortcut=True):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),
            Conv(c_, c_, k, s, p=k // 2, g=c_, act=False) if s == 2 else nn.Identity(),
            GhostConv(c_, c2, 1, 1, act=False),
        )
        self.shortcut = (
            nn.Sequential(
                Conv(c1, c1, k, s, p=k // 2, g=c1, act=False),
                Conv(c1, c2, 1, 1, act=False),
            )
            if s == 2
            else (Conv(c1, c2, 1, 1, act=False) if c1 != c2 else nn.Identity())
        ) if shortcut else None

    def forward(self, x):
        return self.conv(x) + self.shortcut(x) if self.shortcut is not None else self.conv(x)


class C3k2_Ghost(nn.Module):
    """C3k2 block powered by GhostBottleneck v1."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_, 3, 1, shortcut=shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# ==============================================================================
# 2. GHOSTNET-V3 MODULES (WITH STRUCTURAL RE-PARAMETERIZATION)
# ==============================================================================

class RepDWConv(nn.Module):
    """Re-parameterizable Depthwise Conv3x3 for GhostNet-v3 cheap operations.
    
    During training: Multi-branch (3x3 DWConv + 1x1 DWConv + Identity BN).
    During inference: Fused single 3x3 DWConv (Zero Overhead).
    """

    def __init__(self, c, act=True):
        super().__init__()
        self.c = c
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.deploy = False

        # Multi-branch training setup
        self.dw3x3 = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False),
            nn.BatchNorm2d(c)
        )
        self.dw1x1 = nn.Sequential(
            nn.Conv2d(c, c, 1, 1, 0, groups=c, bias=False),
            nn.BatchNorm2d(c)
        )
        self.bn_skip = nn.BatchNorm2d(c)

    def forward(self, x):
        if self.deploy:
            return self.act(self.rbr_reparam(x))
        
        out = self.dw3x3(x) + self.dw1x1(x) + self.bn_skip(x)
        return self.act(out)

    def switch_to_deploy(self):
        if self.deploy:
            return
        w, b = self._get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.c, self.c, 3, 1, 1, groups=self.c, bias=True
        ).to(device=w.device, dtype=w.dtype)
        self.rbr_reparam.weight.data.copy_(w)
        self.rbr_reparam.bias.data.copy_(b)
        self.__delattr__("dw3x3")
        self.__delattr__("dw1x1")
        self.__delattr__("bn_skip")
        self.deploy = True

    def _get_equivalent_kernel_bias(self):
        w3, b3 = self._fuse_bn_tensor(self.dw3x3[0], self.dw3x3[1])
        w1, b1 = self._fuse_bn_tensor(self.dw1x1[0], self.dw1x1[1])
        ws, bs = self._fuse_bn_skip(self.bn_skip)

        # Pad 1x1 kernel to 3x3
        w1_padded = F.pad(w1, (1, 1, 1, 1))

        weight = w3 + w1_padded + ws
        bias = b3 + b1 + bs
        return weight, bias

    def _fuse_bn_tensor(self, conv, bn):
        w = conv.weight
        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight
        beta = bn.bias
        w_fused = w * (gamma / var_sqrt).reshape(-1, 1, 1, 1)
        b_fused = beta - mean * (gamma / var_sqrt)
        return w_fused, b_fused

    def _fuse_bn_skip(self, bn):
        c = self.c
        w = torch.zeros((c, 1, 3, 3), device=bn.weight.device, dtype=bn.weight.dtype)
        w[:, 0, 1, 1] = 1.0
        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight
        beta = bn.bias
        w_fused = w * (gamma / var_sqrt).reshape(-1, 1, 1, 1)
        b_fused = beta - mean * (gamma / var_sqrt)
        return w_fused, b_fused


class GhostConvV3(nn.Module):
    """Ghost Convolution v3 with Re-parameterizable cheap operations."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p=None, g=g, act=act)
        self.cv2 = RepDWConv(c_, act=act)  # Enhanced cheap operation

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), dim=1)


class RepConvN(nn.Module):
    """N-branch Conv-BN block used by the official GhostNetV3 training graph."""

    def __init__(self, c1, c2, k, s=1, p=None, g=1, branches=3, act=True):
        super().__init__()
        self.c1, self.c2, self.k, self.s, self.g = c1, c2, k, s, g
        self.p = k // 2 if p is None else p
        self.deploy = False
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c1, c2, k, s, self.p, groups=g, bias=False),
                nn.BatchNorm2d(c2),
            )
            for _ in range(branches)
        )
        self.scale = (
            nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, 0, groups=g, bias=False),
                nn.BatchNorm2d(c2),
            )
            if k > 1
            else None
        )
        self.skip = nn.BatchNorm2d(c1) if c1 == c2 and s == 1 else None

    def forward(self, x):
        if self.deploy:
            return self.act(self.reparam(x))
        y = sum(branch(x) for branch in self.branches)
        if self.scale is not None:
            y = y + self.scale(x)
        if self.skip is not None:
            y = y + self.skip(x)
        return self.act(y)

    @staticmethod
    def _fuse_conv_bn(conv, bn):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        kernel = conv.weight * scale.reshape(-1, 1, 1, 1)
        bias = bn.bias - bn.running_mean * scale
        return kernel, bias

    def _fuse_skip_bn(self):
        kernel = torch.zeros(
            (self.c2, self.c1 // self.g, self.k, self.k),
            device=self.skip.weight.device,
            dtype=self.skip.weight.dtype,
        )
        center = self.k // 2
        channels_per_group = self.c1 // self.g
        for out_channel in range(self.c2):
            kernel[out_channel, out_channel % channels_per_group, center, center] = 1
        scale = self.skip.weight / torch.sqrt(self.skip.running_var + self.skip.eps)
        kernel = kernel * scale.reshape(-1, 1, 1, 1)
        bias = self.skip.bias - self.skip.running_mean * scale
        return kernel, bias

    def equivalent_kernel_bias(self):
        kernel = 0
        bias = 0
        for branch in self.branches:
            branch_kernel, branch_bias = self._fuse_conv_bn(branch[0], branch[1])
            kernel = kernel + branch_kernel
            bias = bias + branch_bias
        if self.scale is not None:
            scale_kernel, scale_bias = self._fuse_conv_bn(self.scale[0], self.scale[1])
            pad = self.k // 2
            kernel = kernel + F.pad(scale_kernel, (pad, pad, pad, pad))
            bias = bias + scale_bias
        if self.skip is not None:
            skip_kernel, skip_bias = self._fuse_skip_bn()
            kernel = kernel + skip_kernel
            bias = bias + skip_bias
        return kernel, bias

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.equivalent_kernel_bias()
        self.reparam = nn.Conv2d(
            self.c1, self.c2, self.k, self.s, self.p, groups=self.g, bias=True
        ).to(device=kernel.device, dtype=kernel.dtype)
        self.reparam.weight.data.copy_(kernel)
        self.reparam.bias.data.copy_(bias)
        del self.branches
        if self.scale is not None:
            del self.scale
        if self.skip is not None:
            del self.skip
        self.deploy = True


class GhostModuleV3(nn.Module):
    """Standalone GhostNetV3 module adapted from Huawei's official source.

    Training uses three parallel Conv-BN branches on both the primary and cheap
    paths, plus eligible scale and identity branches. Deployment fuses each
    path to one convolution. The DFC gate follows GhostNetV2/V3.
    """

    def __init__(self, c1, c2, k=1, s=1, ratio=2, dw_size=3, act=True):
        super().__init__()
        self.c2 = c2
        intrinsic = math.ceil(c2 / ratio)
        cheap = intrinsic * (ratio - 1)
        self.primary = RepConvN(c1, intrinsic, k, s, branches=3, act=act)
        self.cheap = RepConvN(
            intrinsic, cheap, dw_size, 1, g=intrinsic, branches=3, act=act
        )
        self.gate = nn.Sequential(
            nn.Conv2d(c1, c2, 1, s, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.Conv2d(c2, c2, (1, 5), 1, (0, 2), groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.Conv2d(c2, c2, (5, 1), 1, (2, 0), groups=c2, bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x):
        intrinsic = self.primary(x)
        cheap = self.cheap(intrinsic)
        y = torch.cat((intrinsic, cheap), dim=1)[:, : self.c2]
        pooled = F.avg_pool2d(x, kernel_size=2, stride=2)
        gate = torch.sigmoid(self.gate(pooled))
        gate = F.interpolate(gate, size=y.shape[-2:], mode="nearest")
        return y * gate

    def switch_to_deploy(self):
        self.primary.switch_to_deploy()
        self.cheap.switch_to_deploy()


class GhostBottleneckV3(nn.Module):
    """Ghost Bottleneck v3."""

    def __init__(self, c1, c2, k=3, s=1, shortcut=True):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConvV3(c1, c_, 1, 1),
            Conv(c_, c_, k, s, p=k // 2, g=c_, act=False) if s == 2 else nn.Identity(),
            GhostConvV3(c_, c2, 1, 1, act=False),
        )
        self.shortcut = (
            nn.Sequential(
                Conv(c1, c1, k, s, p=k // 2, g=c1, act=False),
                Conv(c1, c2, 1, 1, act=False),
            )
            if s == 2
            else (Conv(c1, c2, 1, 1, act=False) if c1 != c2 else nn.Identity())
        ) if shortcut else None

    def forward(self, x):
        return self.conv(x) + self.shortcut(x) if self.shortcut is not None else self.conv(x)


class C3k2_GhostV3(nn.Module):
    """C3k2 block powered by GhostBottleneck v3."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(GhostBottleneckV3(c_, c_, 3, 1, shortcut=shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
