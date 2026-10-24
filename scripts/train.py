"""Train HLCQ-DETR from the repository root."""

from ultralytics import RTDETR


if __name__ == "__main__":
    model = RTDETR(
        "framework_patch/ultralytics/cfg/models/rt-detr/hlcq-detr-r18.yaml"
    )
    model.train(
        data="data.yaml",
        cache=False,
        imgsz=640,
        epochs=100,
        batch=4,
        workers=4,
        seed=2,
        project="runs/train",
        name="hlcq-detr",
    )
