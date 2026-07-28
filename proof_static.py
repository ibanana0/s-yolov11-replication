"""Bukti statis S-YOLOv11: jalankan semua verifikasi arsitektur/rumus tanpa training.

Menghasilkan tiap angka yang dikutip di PEMBUKTIAN-REPLIKASI.md. Jalankan:
    python apply_patch.py      # sekali, daftarkan modul kustom ke ultralytics
    python proof_static.py

Output diberi label [B1]..[B7] sesuai section bukti di PEMBUKTIAN-REPLIKASI.md.
"""
import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules_sy import DEConv, WFF, EUCB
from ultralytics.utils.nmiou import nmiou

torch.manual_seed(0)


def hr(tag):
    print(f"\n===== {tag} =====")


# [B1] Backbone: params/GFLOPs baseline YOLOv11s harus cocok paper Tabel 6 (9.4M/21.3).
hr("[B1] baseline YOLOv11s .info()")
base = YOLO("yolo11s.yaml")
base.model.nc = 10
base.model.info()  # cetak: "YOLO11s summary: ... 9,458,752 parameters, 21.7 GFLOPs"

# [B7] Model penuh: params/GFLOPs vs paper Tabel 4/6 (12.1M/36.1).
hr("[B7] s-yolov11 .info()")
m = YOLO("s-yolov11.yaml")
m.model.info()  # cetak: "s-YOLOv11 summary: ... 11,197,427 parameters, 40.0 GFLOPs"

# [B2] EMAFPN: forward 640x640 harus keluar 4 head di skala 160/80/40/20 (paper §2.1 + Fig.2).
hr("[B2] forward output scales")
m.model.eval()
with torch.no_grad():
    out = m.model(torch.randn(1, 3, 640, 640))
feats = out[1] if isinstance(out, tuple) else out
print("N_HEADS", len(feats), "GRID_SIZES", [tuple(f.shape[-2:]) for f in feats])
print("CH_PER_HEAD", [f.shape[1] for f in feats], "(=4*reg_max+nc=4*16+10=74)")

# [B5] ESDCDH: blok share/cv_reg/cv_cls satu instansi dipakai 4 level (paper §2.3 + Fig.5).
hr("[B5] head param sharing")
det = m.model.model[-1]
print("HEAD_CLASS", type(det).__name__, "nl", det.nl, "reg_max", det.reg_max, "nc", det.nc)
percv = [{id(p) for p in c.parameters()} for c in det.conv]
print("N_SHARE_PARAMS", len(list(det.share.parameters())),
      "N_REG", len(list(det.cv_reg.parameters())),
      "N_CLS", len(list(det.cv_cls.parameters())))
print("PER_LEVEL_CGS_DISTINCT",
      all(percv[i].isdisjoint(percv[j]) for i in range(len(percv)) for j in range(i + 1, len(percv))))
print("N_SCALES", len(det.scales), "scale_vals", [round(s.scale.item(), 2) for s in det.scales])

# [B4] DEConv: 5 cabang, conv gabungan == jumlah 5 cabang (paper §2.3 reparam).
hr("[B4] DEConv reparam")
c = 8
de = DEConv(c).eval()
x = torch.randn(1, c, 16, 16)
with torch.no_grad():
    fused = de(x)
    parts = [de.cd, de.hd, de.vd, de.ad]
    ws = [p.get_weight()[0] for p in parts] + [de.std.weight]
    bs = [p.get_weight()[1] for p in parts] + [de.std.bias]
    explicit = sum(F.conv2d(x, w, b, padding=1) for w, b in zip(ws, bs))
print("DECONV_N_BRANCHES", len(ws))
print("DECONV_REPARAM_MAXDIFF", (fused - explicit).abs().max().item())

# [B3] WFF: bobot >=0, sum~1, output == rumus paper §2.2 (persamaan 68).
hr("[B3] WFF formula + EUCB upsample")
w = WFF(3).eval()
a, b3, c3 = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
with torch.no_grad():
    wn = torch.relu(w.w)
    wn = wn / (wn.sum() + w.eps)
    manual = wn[0] * a + wn[1] * b3 + wn[2] * c3
    got = w([a, b3, c3])
print("WFF_WEIGHTS_NONNEG", bool((wn >= 0).all()), "WFF_SUM~1", round(wn.sum().item(), 5))
print("WFF_MATCHES_FORMULA_MAXDIFF", (got - manual).abs().max().item())
e = EUCB(8, 4).eval()
print("EUCB_OUT_SHAPE", tuple(e(torch.randn(1, 8, 10, 10)).shape))

# [B6] NMIoU: identik->1, monoton, sweep alpha (paper §2.4 + Tabel 3).
hr("[B6] NMIoU behaviour")
b = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
far = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
print("NMIOU_IDENTICAL", round(nmiou(b, b).item(), 4), "SHAPE", tuple(nmiou(b, b).shape))
print("MONOTONIC_dist1>dist5", nmiou(b, b + 1).item() > nmiou(b, b + 5).item())
print("NMIOU_FAR", round(nmiou(b, far).item(), 4))
for al in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    print(f"  alpha={al}: nmiou(b,b+3)={nmiou(b, b + 3, alpha=al).item():.4f}")
