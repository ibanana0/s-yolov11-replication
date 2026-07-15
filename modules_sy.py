# Modul S-YOLOv11 (Wang et al., 2025) — komponen EMAFPN + Detect_ESDCDH.
# Salin file ini ke ultralytics/nn/modules_sy.py, lalu ikuti PATCH-ULTRALYTICS.md.
#
# Sumber (lihat replikasi/REFERENCES.md):
# - WFF        : BiFPN, https://github.com/google/automl/tree/master/efficientdet
# - EUCB       : EMCAD (Rahman et al., CVPR 2024)
# - DEConv dkk : DEA-Net, https://github.com/cecret3350/DEA-Net (code/model/modules/deconv.py)
#                — diverifikasi identik dengan repo resmi (permutasi ad, layout hd/vd Conv1d,
#                penjumlahan 5 bobot menjadi satu conv2d), tanpa dependensi einops/CUDA-only.
# - Scale      : FCOS, https://github.com/tianzhi0549/FCOS
# - Topologi EMAFPN: rekonstruksi dari persamaan P_td/P_out paper (Fig. 2-4);
#                MAFPN asli (https://github.com/yang-0201/MAF-YOLO) berbasis YOLOv6,
#                tidak bisa dipakai langsung.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.head import Detect


class WFF(nn.Module):
    """Weighted Feature Fusion (gaya BiFPN): O = sum(w_i * I_i) / (eps + sum(w_j)), w_i >= 0 via ReLU."""

    def __init__(self, n=2):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n))
        self.eps = 1e-4

    def forward(self, x):
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        return sum(w[i] * x[i] for i in range(len(x)))


class EUCB(nn.Module):
    """Efficient Up-Convolution Block (EMCAD): upsample 2x -> depthwise 3x3 -> pointwise 1x1."""

    def __init__(self, c1, c2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dw = Conv(c1, c1, 3, g=c1)
        self.pw = Conv(c1, c2, 1)

    def forward(self, x):
        return self.pw(self.dw(self.up(x)))


# ---------- DEConv (DEA-Net, Chen et al. 2024): 5 cabang konvolusi paralel ----------


class Conv2d_cd(nn.Module):
    """Central difference convolution: bobot pusat dikurangi jumlah seluruh bobot."""

    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, padding=1, bias=True)

    def get_weight(self):
        w = self.conv.weight
        w_cd = w.clone()
        w_cd[:, :, 1, 1] = w[:, :, 1, 1] - w.sum(dim=(2, 3))
        return w_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    """Angular difference convolution: selisih bobot dengan permutasi rotasi tetangga."""

    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, padding=1, bias=True)

    def get_weight(self):
        w = self.conv.weight.flatten(2)
        w_ad = (w - w[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view_as(self.conv.weight)
        return w_ad, self.conv.bias


class Conv2d_hd(nn.Module):
    """Horizontal difference convolution: kolom kiri positif, kolom kanan negatif."""

    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, 3, padding=1, bias=True)

    def get_weight(self):
        w = self.conv.weight  # (c2, c1, 3)
        w_hd = w.new_zeros(w.shape[0], w.shape[1], 9)
        w_hd[:, :, [0, 3, 6]] = w
        w_hd[:, :, [2, 5, 8]] = -w
        return w_hd.view(w.shape[0], w.shape[1], 3, 3), self.conv.bias


class Conv2d_vd(nn.Module):
    """Vertical difference convolution: baris atas positif, baris bawah negatif."""

    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, 3, padding=1, bias=True)

    def get_weight(self):
        w = self.conv.weight
        w_vd = w.new_zeros(w.shape[0], w.shape[1], 9)
        w_vd[:, :, [0, 1, 2]] = w
        w_vd[:, :, [6, 7, 8]] = -w
        return w_vd.view(w.shape[0], w.shape[1], 3, 3), self.conv.bias


class DEConv(nn.Module):
    """Detail-Enhanced Convolution: 5 cabang 3x3 digabung ekuivalen jadi satu konvolusi.

    ponytail: bobot dijumlah tiap forward (ekuivalen reparameterisasi paper);
    tambahkan switch_to_deploy hanya jika latency inference jadi masalah.
    """

    def __init__(self, c):
        super().__init__()
        self.cd = Conv2d_cd(c, c)
        self.hd = Conv2d_hd(c, c)
        self.vd = Conv2d_vd(c, c)
        self.ad = Conv2d_ad(c, c)
        self.std = nn.Conv2d(c, c, 3, padding=1, bias=True)

    def forward(self, x):
        ws, bs = zip(
            self.cd.get_weight(),
            self.hd.get_weight(),
            self.vd.get_weight(),
            self.ad.get_weight(),
            (self.std.weight, self.std.bias),
        )
        return F.conv2d(x, sum(ws), sum(bs), padding=1)


# ---------- Detection head ----------


class ConvGN(nn.Module):
    """Conv + GroupNorm + SiLU (pengganti Conv+BN pada head, mengikuti paper/FCOS)."""

    def __init__(self, c1, c2, k=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, padding=k // 2, bias=False)
        self.gn = nn.GroupNorm(16, c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class DECGS(nn.Module):
    """DEConv + GroupNorm + SiLU (blok shared conv pada ESDCDH)."""

    def __init__(self, c):
        super().__init__()
        self.conv = DEConv(c)
        self.gn = nn.GroupNorm(16, c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class Scale(nn.Module):
    """Skalar learnable per level untuk menangani variasi ukuran objek antar skala."""

    def __init__(self, s=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(s))

    def forward(self, x):
        return x * self.scale


class Detect_ESDCDH(Detect):
    """Efficient Shared Detail-Enhanced Convolutional Detection Head.

    Per level: ConvGN 1x1 -> 2x DECGS (parameter shared antar level)
    -> cabang reg (Scale per level) + cabang cls (shared).
    """

    def __init__(self, nc=80, hidc=128, ch=()):
        super().__init__(nc, ch)
        del self.cv2, self.cv3  # cabang default Detect diganti cabang shared
        self.conv = nn.ModuleList(ConvGN(c, hidc, 1) for c in ch)
        self.share = nn.Sequential(DECGS(hidc), DECGS(hidc))
        self.cv_reg = nn.Conv2d(hidc, 4 * self.reg_max, 3, padding=1)
        self.cv_cls = nn.Conv2d(hidc, self.nc, 3, padding=1)
        self.scales = nn.ModuleList(Scale(1.0) for _ in ch)

    def forward(self, x):
        for i in range(self.nl):
            f = self.share(self.conv[i](x[i]))
            x[i] = torch.cat((self.scales[i](self.cv_reg(f)), self.cv_cls(f)), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        # ponytail: head shared -> satu bias untuk semua level, pakai stride tengah 16
        self.cv_reg.bias.data[:] = 1.0
        self.cv_cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / 16) ** 2)
