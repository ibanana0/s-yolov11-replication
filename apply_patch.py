#!/usr/bin/env python3
"""Patch an editable Ultralytics install for S-YOLOv11 (PATCH-ULTRALYTICS.md steps 2-4).

Idempotent: safe to re-run. Auto-detects the installed ultralytics package.
Used both on the host (sanity check) and inside the Docker image (training).

Run:  python replikasi/apply_patch.py
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
import ultralytics  # noqa: E402

PKG = Path(ultralytics.__file__).resolve().parent
print(f"ultralytics {ultralytics.__version__} at {PKG}")


def replace_once(path: Path, old: str, new: str, tag: str):
    # utf-8 eksplisit: default Windows (cp1252) gagal baca sumber ultralytics.
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"  [skip] {tag} already applied")
        return
    if old not in text:
        raise SystemExit(f"  [FAIL] anchor for {tag} not found in {path} (version mismatch?)")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"  [ok]   {tag}")


# --- step 2: copy custom modules ---
shutil.copy(HERE / "modules_sy.py", PKG / "nn" / "modules_sy.py")
shutil.copy(HERE / "nmiou.py", PKG / "utils" / "nmiou.py")
print("  [ok]   copied modules_sy.py, nmiou.py")

# --- step 3: patch nn/tasks.py ---
tasks = PKG / "nn" / "tasks.py"

# 3a. import
replace_once(
    tasks,
    "from ultralytics.nn.modules import (",
    "from ultralytics.nn.modules_sy import EUCB, WFF, Detect_ESDCDH\n"
    "from ultralytics.nn.modules import (",
    "tasks: import",
)

# 3b. register EUCB as a channelled module (gets (c1, c2) like Conv)
replace_once(
    tasks,
    "        if m in {\n            Classify,\n            Conv,\n",
    "        if m in {\n            Classify,\n            Conv,\n            EUCB,\n",
    "tasks: EUCB in channel set",
)

# 3c. WFF branch (multi-input, channels unchanged) — insert before final else
replace_once(
    tasks,
    "        elif m is CBFuse:\n            c2 = ch[f[-1]]\n",
    "        elif m is CBFuse:\n            c2 = ch[f[-1]]\n"
    "        elif m is WFF:\n            c2 = ch[f[0]]\n",
    "tasks: WFF branch",
)

# 3d. register Detect_ESDCDH as a detection head
replace_once(
    tasks,
    "        elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}:",
    "        elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect, Detect_ESDCDH}:",
    "tasks: Detect_ESDCDH in head set",
)

# --- step 4: patch utils/loss.py (NWD-MPD-IoU) ---
loss = PKG / "utils" / "loss.py"
replace_once(
    loss,
    "from .metrics import bbox_iou, probiou",
    "from .metrics import bbox_iou, probiou\nfrom .nmiou import nmiou",
    "loss: import nmiou",
)
replace_once(
    loss,
    "iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)",
    "iou = nmiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], alpha=0.8)",
    "loss: swap CIoU -> nmiou",
)

print("patch complete")
