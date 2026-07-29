"""Train one controlled S-YOLOv11 GhostNetV3/DWConv ablation.

Examples:
    python replikasi/train_ablation.py --variant baseline
    python replikasi/train_ablation.py --variant ghostv3 --device 0
    python replikasi/train_ablation.py --variant all --device 0
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "baseline": ROOT / "s-yolov11.yaml",
    "ghostv3": ROOT / "s-yolov11-ghostmodulev3.yaml",
    "dwconv": ROOT / "s-yolov11-dwconv.yaml",
    "ghostv3_dwconv": ROOT / "s-yolov11-ghostmodulev3-dwconv.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], required=True)
    parser.add_argument("--data", default="VisDrone.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/s-yolov11-ablation")
    return parser.parse_args()


def train_variant(name, cfg, args):
    model = YOLO(str(cfg))
    model.info()
    return model.train(
        data=args.data,
        epochs=args.epochs,
        patience=100,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        optimizer="SGD",
        close_mosaic=10,
        warmup_epochs=3.0,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=f"{name}-seed{args.seed}",
        exist_ok=False,
    )


def main():
    args = parse_args()
    selected = VARIANTS.items() if args.variant == "all" else [(args.variant, VARIANTS[args.variant])]
    for name, cfg in selected:
        train_variant(name, cfg, args)


if __name__ == "__main__":
    main()
