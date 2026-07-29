"""Build and compare all controlled S-YOLOv11 GhostNetV3/DWConv variants."""

import torch

from ultralytics import YOLO
from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.conv import DWConv
from ultralytics.nn.modules_ghost import GhostModuleV3
from ultralytics.utils.torch_utils import get_flops

from train_ablation import VARIANTS


EXPECTED = {
    "baseline": (11_197_427, 0, 0),
    "ghostv3": (11_024_883, 1, 0),
    "dwconv": (9_270_707, 0, 5),
    "ghostv3_dwconv": (9_098_163, 1, 5),
}


def features(output):
    return output[1] if isinstance(output, tuple) else output


def main():
    torch.manual_seed(0)
    rows = {}
    for name, cfg in VARIANTS.items():
        model = YOLO(str(cfg)).model
        params = sum(parameter.numel() for parameter in model.parameters())
        ghosts = sum(isinstance(module, GhostModuleV3) for module in model.modules())
        backbone_dw = sum(isinstance(model.model[index], DWConv) for index in (0, 1, 3, 5, 7))
        assert (params, ghosts, backbone_dw) == EXPECTED[name]
        assert sum(isinstance(module, C3k2) for module in model.modules()) == 12 - ghosts

        model.eval()
        with torch.no_grad():
            output = features(model(torch.randn(1, 3, 128, 128)))
        grids = [tuple(tensor.shape[-2:]) for tensor in output]
        assert grids == [(32, 32), (16, 16), (8, 8), (4, 4)]
        rows[name] = (params, get_flops(model, imgsz=640), grids)

    reduction = rows["baseline"][0] - rows["dwconv"][0]
    assert reduction == 1_926_720
    print("All ablation models passed")
    for name, (params, flops, _) in rows.items():
        print(f"{name:16s} params={params:,} GFLOPs={flops:.4f}")
    print("DWConv parameter reduction", f"{reduction:,}")


if __name__ == "__main__":
    main()
