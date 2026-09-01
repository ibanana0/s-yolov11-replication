"""Evaluate the three ablation checkpoints on VisDrone test-dev."""

from __future__ import annotations

import argparse
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIRECTORY = ROOT / "hasil-s-yolov11" / "s-yolov11-main"
DEFAULT_MODEL_PATHS = {
    "ghostv3": DEFAULT_WEIGHTS_DIRECTORY / "ghostv3-seed0" / "weights" / "best.pt",
    "dwconv": DEFAULT_WEIGHTS_DIRECTORY / "dwconv-seed0" / "weights" / "best.pt",
    "ghostv3_dwconv": (
        DEFAULT_WEIGHTS_DIRECTORY
        / "ghostv3_dwconv-seed0"
        / "weights"
        / "best.pt"
    ),
}


class _Tee:
    """Write evaluator output to both the terminal and a model log."""

    def __init__(self, console: Any, log_stream: Any):
        self.console = console
        self.log_stream = log_stream

    def write(self, value: str) -> int:
        self.console.write(value)
        self.log_stream.write(value)
        return len(value)

    def flush(self) -> None:
        self.console.flush()
        self.log_stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.console, "isatty", lambda: False)())


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _status(queue_log: Path, message: str) -> str:
    line = f"[{_timestamp()}] {message}"
    print(line, flush=True)
    with queue_log.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    return line


def _metrics_dict(metrics: Any) -> Any:
    return getattr(metrics, "results_dict", metrics)


def evaluate_variants(
    model_paths: Mapping[str, Path | str] = DEFAULT_MODEL_PATHS,
    *,
    data: str = "VisDrone.yaml",
    batch: int = 4,
    imgsz: int = 640,
    device: str | None = "0",
    project: str | Path = "runs/s-yolov11-test",
    log_directory: str | Path = ROOT / "logs" / "test",
    model_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Evaluate each checkpoint on the test split and persist all output logs."""

    if model_loader is None:
        from ultralytics import YOLO

        model_loader = YOLO

    log_directory = Path(log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)
    queue_log = log_directory / "queue.log"
    project = str(project)
    results: dict[str, Any] = {}

    _status(
        queue_log,
        f"TEST EVALUATION START models={','.join(model_paths)} split=test",
    )

    for name, model_path in model_paths.items():
        model_path = Path(model_path)
        model_log = log_directory / f"{name}.log"
        start_line = _status(queue_log, f"START {name}; weights={model_path}")

        with model_log.open("w", encoding="utf-8") as log_stream:
            log_stream.write(start_line + "\n")
            log_stream.flush()
            console_stdout = sys.stdout
            tee = _Tee(console_stdout, log_stream)
            try:
                with redirect_stdout(tee), redirect_stderr(tee):
                    print(f"weights={model_path}")
                    print(
                        f"data={data} split=test imgsz={imgsz} "
                        f"batch={batch} device={device} project={project}"
                    )
                    model = model_loader(str(model_path))
                    metrics = model.val(
                        data=data,
                        split="test",
                        imgsz=imgsz,
                        batch=batch,
                        device=device,
                        plots=True,
                        project=project,
                        name=name,
                        exist_ok=True,
                    )
                    print(f"metrics={_metrics_dict(metrics)}")
                    results[name] = metrics
            except Exception:
                traceback.print_exc()
                _status(queue_log, f"FAILED {name}; log={model_log}")
                raise

        _status(queue_log, f"DONE {name}; log={model_log}")

    _status(queue_log, "TEST EVALUATION COMPLETE")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the GhostV3/DWConv ablation checkpoints on VisDrone test-dev."
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=DEFAULT_WEIGHTS_DIRECTORY,
        help="Directory containing <variant>-seed0/weights/best.pt.",
    )
    parser.add_argument("--data", default="VisDrone.yaml")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/s-yolov11-test")
    parser.add_argument("--log-directory", type=Path, default=ROOT / "logs" / "test")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_paths = {
        name: args.weights_dir / f"{name}-seed0" / "weights" / "best.pt"
        for name in DEFAULT_MODEL_PATHS
    }
    evaluate_variants(
        model_paths=model_paths,
        data=args.data,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        log_directory=args.log_directory,
    )


if __name__ == "__main__":
    main()
