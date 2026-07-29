"""Run S-YOLOv11 ablations sequentially with one log per model.

The next variant starts only after the current variant exits successfully.
The queue stops immediately if a training process fails.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from train_ablation import VARIANTS


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(VARIANTS),
        help="Variants and execution order. Default: all four variants.",
    )
    parser.add_argument("--data", default="VisDrone.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/s-yolov11-main")
    parser.add_argument("--log-root", default="logs/ablation")
    return parser.parse_args()


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_status(queue_log, message):
    line = f"[{timestamp()}] {message}"
    print(line, flush=True)
    with queue_log.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def run_variant(variant, args, log_dir, queue_log):
    model_log = log_dir / f"{variant}.log"
    command = [
        sys.executable,
        "-u",
        str(ROOT / "train_ablation.py"),
        "--variant",
        variant,
        "--data",
        args.data,
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--imgsz",
        str(args.imgsz),
        "--workers",
        str(args.workers),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--project",
        args.project,
    ]
    write_status(queue_log, f"START {variant}; log={model_log}")
    with model_log.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_stream.write(line)
                log_stream.flush()
        except KeyboardInterrupt:
            process.send_signal(2)
            process.wait()
            write_status(queue_log, f"INTERRUPTED {variant}; exit={process.returncode}")
            raise
        return_code = process.wait()
    if return_code != 0:
        write_status(queue_log, f"FAILED {variant}; exit={return_code}; queue stopped")
        raise SystemExit(return_code)
    write_status(queue_log, f"DONE {variant}")


def main():
    args = parse_args()
    log_dir = ROOT / args.log_root / f"seed{args.seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue_log = log_dir / "queue.log"

    write_status(
        queue_log,
        f"QUEUE START variants={','.join(args.variants)} epochs={args.epochs} seed={args.seed}",
    )
    for variant in args.variants:
        run_variant(variant, args, log_dir, queue_log)
    write_status(queue_log, "QUEUE COMPLETE")


if __name__ == "__main__":
    main()
