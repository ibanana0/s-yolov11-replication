"""Training S-YOLOv11 di VisDrone2019.

Dataset VisDrone di-download + dikonversi otomatis oleh Ultralytics (VisDrone.yaml bawaan).
Hyperparameter mengikuti Tabel 1 paper.
"""

from pathlib import Path

from ultralytics import YOLO


# main-guard wajib di Windows: dataloader workers>0 pakai spawn -> re-import modul.
# Tanpa ini training akan rekursif. (ponytail: guard, bukan workers=0, agar tetap cepat.)
def main():
    model = YOLO(str(Path(__file__).parent / "s-yolov11.yaml"))
    model.info()  # cek sanity: target paper ~12.1M params, ~36.1 GFLOPs

    model.train(
        data="VisDrone.yaml",
        epochs=300,
        patience=100,
        batch=8,
        imgsz=640,
        workers=4,
        optimizer="SGD",
        close_mosaic=10,
        warmup_epochs=3.0,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
    )


if __name__ == "__main__":
    main()
