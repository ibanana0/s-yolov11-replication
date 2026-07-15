# S-YOLOv11 replication image. Build + run on a CUDA GPU box (NOT the M1 Mac).
# Base pins paper's stack: torch 2.0.0 / CUDA 11.8.
FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime

# libGL/glib needed by opencv (ultralytics dep)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Ultralytics clone lives OUTSIDE the workdir so `import ultralytics` never gets
# shadowed by the ./ultralytics folder (the bug that bit the host run).
WORKDIR /opt
RUN git clone --depth 1 --branch v8.3.0 https://github.com/ultralytics/ultralytics.git
RUN pip install --no-cache-dir -e /opt/ultralytics

# Custom modules + patch script.
WORKDIR /workspace
COPY modules_sy.py nmiou.py apply_patch.py s-yolov11.yaml train.py ./
# apply_patch.py expects modules_sy.py / nmiou.py next to it -> workspace has them.
RUN python apply_patch.py && python nmiou.py

# Heavy + mutable data on volumes, not in the image.
RUN yolo settings datasets_dir=/data/datasets runs_dir=/data/runs
VOLUME ["/data"]

# Default: sanity info. Override with train.py to train.
CMD ["python", "-c", "from ultralytics import YOLO; YOLO('s-yolov11.yaml').info()"]
