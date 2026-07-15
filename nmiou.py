# Loss NWD-MPD-IoU (NMIoU) untuk S-YOLOv11. Semua box format xyxy.
# Salin ke ultralytics/utils/nmiou.py, lalu ikuti PATCH-ULTRALYTICS.md.
#
# Sumber (lihat replikasi/REFERENCES.md):
# - NWD    : Wang et al. 2021, https://github.com/jwwangchn/NWD (wasserstein loss, mmdet)
# - MPDIoU : Ma & Xu 2023, arXiv:2307.07662 — tidak ada repo resmi, implementasi manual
#            dari rumus paper section 2.4.
import torch


def nwd(box1, box2, C=12.8, eps=1e-7):
    """Normalized Wasserstein Distance (Wang et al., 2021).

    Paper S-YOLOv11 menulis exp(-W2^2/C); versi asli NWD memakai exp(-sqrt(W2^2)/C).
    Dipakai versi asli (sqrt) karena lebih stabil dan hampir pasti yang dimaksud.
    """
    cx1, cy1 = (box1[..., 0] + box1[..., 2]) / 2, (box1[..., 1] + box1[..., 3]) / 2
    cx2, cy2 = (box2[..., 0] + box2[..., 2]) / 2, (box2[..., 1] + box2[..., 3]) / 2
    w1, h1 = box1[..., 2] - box1[..., 0], box1[..., 3] - box1[..., 1]
    w2, h2 = box2[..., 2] - box2[..., 0], box2[..., 3] - box2[..., 1]
    d2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
    return torch.exp(-d2.clamp(min=eps).sqrt() / C)


def mpdiou(box1, box2, eps=1e-7):
    """MPDIoU (Ma & Xu, 2023).

    ponytail: paper menormalkan d1,d2 dengan (w,h) citra input; di sini dipakai diagonal
    enclosing box karena koordinat loss Ultralytics berskala grid per level. Perilaku
    monoton sama; ganti ke dimensi citra jika ingin fidelitas penuh.
    """
    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    a1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    a2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    iou = inter / (a1 + a2 - inter + eps)
    d1 = (box2[..., 0] - box1[..., 0]) ** 2 + (box2[..., 1] - box1[..., 1]) ** 2
    d2 = (box2[..., 2] - box1[..., 2]) ** 2 + (box2[..., 3] - box1[..., 3]) ** 2
    cw = torch.max(box1[..., 2], box2[..., 2]) - torch.min(box1[..., 0], box2[..., 0])
    ch = torch.max(box1[..., 3], box2[..., 3]) - torch.min(box1[..., 1], box2[..., 1])
    diag2 = cw**2 + ch**2 + eps
    return iou - d1 / diag2 - d2 / diag2


def nmiou(box1, box2, alpha=0.8, C=12.8):
    """NMIoU = alpha * MPDIoU + (1 - alpha) * NWD. alpha=0.8 terbaik menurut Tabel 3 paper."""
    return alpha * mpdiou(box1, box2) + (1 - alpha) * nwd(box1, box2, C)


if __name__ == "__main__":
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert torch.allclose(nmiou(b, b), torch.ones(1), atol=1e-3), "box identik harus ~1"
    far = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    assert nmiou(b, far).item() < 0.1, "box jauh harus mendekati/kurang dari 0"
    assert nmiou(b, b + 1.0).item() > nmiou(b, b + 5.0).item(), "harus monoton terhadap jarak"
    print("nmiou OK")
