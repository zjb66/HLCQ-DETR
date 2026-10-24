"""Validate a trained HLCQ-DETR checkpoint from the repository root."""

from ultralytics import RTDETR


if __name__ == "__main__":
    model = RTDETR("weights/best.pt")
    model.val(
        data="data.yaml",
        split="test",
        imgsz=640,
        batch=4,
        workers=4,
        project="runs/val",
        name="hlcq-detr",
    )
