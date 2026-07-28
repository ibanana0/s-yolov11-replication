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

        # Deploy conv (used after switch_to_deploy)
        self.rbr_reparam = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=True)

    def forward(self, x):
        if self.deploy:
            return self.act(self.rbr_reparam(x))
        
        out = self.dw3x3(x) + self.dw1x1(x) + self.bn_skip(x)
        return self.act(out)

    def switch_to_deploy(self):
        if self.deploy:
            return
        w, b = self._get_equivalent_kernel_bias()
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
