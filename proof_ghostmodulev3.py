"""Static checks for the standalone GhostModuleV3 S-YOLOv11 ablation."""

from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules_ghost import GhostModuleV3, RepConvN


def detection_features(output):
    return output[1] if isinstance(output, tuple) else output


def main():
    torch.manual_seed(0)
    cfg = Path(__file__).parent / "s-yolov11-ghostmodulev3.yaml"
    model = YOLO(str(cfg)).model
    ghost_modules = [module for module in model.modules() if isinstance(module, GhostModuleV3)]

    assert len(ghost_modules) == 1
    assert sum(isinstance(module, C3k2) for module in model.modules()) == 11
    assert type(model.model[6]).__name__ == "GhostModuleV3"
    assert all(type(model.model[index]).__name__ == "C3k2" for index in (2, 4, 8))
    assert all(
        type(model.model[index]).__name__ == "C3k2"
        for index in (16, 20, 24, 28, 30, 34, 38, 42)
    )

    model.eval()
    sample = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        before = detection_features(model(sample))
    assert [tuple(tensor.shape[-2:]) for tensor in before] == [
        (32, 32),
        (16, 16),
        (8, 8),
        (4, 4),
    ]

    model.train()
    output = model(torch.randn(2, 3, 64, 64))
    sum(tensor.mean() for tensor in output).backward()
    missing = [
        name
        for name, parameter in ghost_modules[0].named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing, f"GhostModuleV3 parameters without gradients: {missing}"

    model.eval()
    with torch.no_grad():
        before = detection_features(model(sample))
    ghost_modules[0].switch_to_deploy()
    with torch.no_grad():
        after = detection_features(model(sample))
    max_diff = max(
        (left - right).abs().max().item() for left, right in zip(before, after)
    )
    assert max_diff < 1e-4, f"Train/deploy fusion mismatch: {max_diff}"
    assert all(
        module.deploy
        for module in ghost_modules[0].modules()
        if isinstance(module, RepConvN)
    )

    params = sum(parameter.numel() for parameter in model.parameters())
    print("GhostModuleV3 checks passed")
    print("deploy_maxdiff", max_diff)
    print("post_deploy_params", params)


if __name__ == "__main__":
    main()
