from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou, nms
from torchvision.transforms.functional import pil_to_tensor


def build_knee_detector(*, pretrained: bool = False) -> torch.nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None if not pretrained else None,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


def load_detector_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    state = _upgrade_legacy_detector_state_dict(state)
    model = build_knee_detector(pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, checkpoint.get("metadata", {})


def _upgrade_legacy_detector_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map torchvision 0.8-era FPN/RPN names to current module names."""
    upgraded = {}
    for key, value in state.items():
        new_key = key
        for block in ("inner_blocks", "layer_blocks"):
            prefix = f"backbone.fpn.{block}."
            if key.startswith(prefix):
                remainder = key[len(prefix) :]
                index, separator, suffix = remainder.partition(".")
                if separator and suffix in {"weight", "bias"}:
                    new_key = f"{prefix}{index}.0.{suffix}"
        if key == "rpn.head.conv.weight":
            new_key = "rpn.head.conv.0.0.weight"
        elif key == "rpn.head.conv.bias":
            new_key = "rpn.head.conv.0.0.bias"
        upgraded[new_key] = value
    return upgraded


def detector_tensor(image: Image.Image) -> torch.Tensor:
    gray_rgb = image.convert("L").convert("RGB")
    return pil_to_tensor(gray_rgb).to(torch.float32) / 255.0


def select_bilateral_detections(
    prediction: dict[str, torch.Tensor],
    *,
    score_threshold: float = 0.5,
    nms_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    keep = scores >= float(score_threshold)
    boxes, scores = boxes[keep], scores[keep]
    if boxes.numel() == 0:
        raise RuntimeError("The detector returned no knee above the score threshold.")
    retained = nms(boxes, scores, float(nms_threshold))
    boxes, scores = boxes[retained], scores[retained]
    order = torch.argsort(scores, descending=True)[:2]
    boxes, scores = boxes[order], scores[order]
    if len(boxes) != 2:
        raise RuntimeError(f"Expected two knees, but detected {len(boxes)}.")
    return boxes, scores


def square_crop_box(
    box: torch.Tensor | np.ndarray,
    image_size: tuple[int, int],
    *,
    margin_pixels: int = 10,
) -> tuple[int, int, int, int]:
    """Construct the square ROI from predicted horizontal bounds and centre."""
    x1, y1, x2, y2 = map(float, box)
    centre_y = (y1 + y2) / 2.0
    side = max(x2 - x1, 1.0)
    x1 -= margin_pixels
    x2 += margin_pixels
    y1 = centre_y - side / 2.0 - margin_pixels
    y2 = centre_y + side / 2.0 + margin_pixels
    width, height = image_size
    return (
        max(0, int(np.floor(x1))),
        max(0, int(np.floor(y1))),
        min(width, int(np.ceil(x2))),
        min(height, int(np.ceil(y2))),
    )


def detect_and_crop_bilateral(
    model: torch.nn.Module,
    image: Image.Image,
    *,
    device: str | torch.device,
    score_threshold: float,
    nms_threshold: float,
    margin_pixels: int,
    patient_right_on_image_left: bool,
) -> dict[str, dict[str, Any]]:
    tensor = detector_tensor(image).to(device)
    model.eval()
    with torch.inference_mode():
        prediction = model([tensor])[0]
    boxes, scores = select_bilateral_detections(
        prediction,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
    )
    order = torch.argsort((boxes[:, 0] + boxes[:, 2]) / 2.0)
    image_left, image_right = order.tolist()
    if patient_right_on_image_left:
        side_to_index = {"right": image_left, "left": image_right}
    else:
        side_to_index = {"left": image_left, "right": image_right}
    result = {}
    for side, index in side_to_index.items():
        crop_box = square_crop_box(
            boxes[index], image.size, margin_pixels=margin_pixels
        )
        result[side] = {
            "detection_box": tuple(map(float, boxes[index].tolist())),
            "crop_box": crop_box,
            "confidence": float(scores[index]),
            "crop": image.crop(crop_box),
        }
    return result


def _labelme_boxes(path: Path) -> torch.Tensor:
    with path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)
    boxes = []
    for shape in annotation.get("shapes", []):
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
    if len(boxes) != 2:
        raise ValueError(f"Expected two rectangular knee boxes in {path}, found {len(boxes)}")
    return torch.tensor(boxes, dtype=torch.float32)


class DetectorDataset(Dataset):
    """Participant-disjoint detector metadata with LabelMe rectangle annotations."""

    def __init__(self, metadata: pd.DataFrame) -> None:
        required = {"participant_id", "cohort", "image_path", "annotation_path", "split"}
        missing = required.difference(metadata.columns)
        if missing:
            raise ValueError(f"Detector metadata is missing: {sorted(missing)}")
        self.metadata = metadata.reset_index(drop=True).copy()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        with Image.open(row["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("L").convert("RGB")
        boxes = _labelme_boxes(Path(row["annotation_path"]))
        labels = torch.ones(len(boxes), dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return detector_tensor(image), target


def detector_collate(batch):
    return tuple(zip(*batch))


def train_detector(
    metadata: pd.DataFrame,
    output_path: str | Path,
    *,
    device: str | torch.device | None = None,
    epochs: int = 25,
    batch_size: int = 8,
    learning_rate: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    scheduler_gamma: float = 0.9,
    scheduler_step_size: int = 3,
    num_workers: int = 0,
) -> Path:
    train_metadata = metadata[metadata["split"].str.lower().eq("train")].copy()
    test_metadata = metadata[metadata["split"].str.lower().eq("test")].copy()
    overlap = set(train_metadata["participant_id"]) & set(test_metadata["participant_id"])
    if overlap:
        raise ValueError("Detector train/test participant leakage was detected.")
    model = build_knee_detector(pretrained=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    loader = DataLoader(
        DetectorDataset(train_metadata),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=detector_collate,
    )
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
    )
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        scheduler.step()
        print(f"Detector epoch {epoch:02d}/{epochs}: loss={epoch_loss / len(loader):.5f}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": {
                "format_version": 1,
                "architecture": "fasterrcnn_resnet50_fpn",
                "classes": ["background", "knee"],
                "epochs": epochs,
                "batch_size": batch_size,
                "optimizer": "SGD",
                "learning_rate": learning_rate,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "scheduler": "StepLR",
                "scheduler_step_size": scheduler_step_size,
                "scheduler_gamma": scheduler_gamma,
            },
        },
        output_path,
    )
    return output_path


def evaluate_detector(
    model: torch.nn.Module,
    metadata: pd.DataFrame,
    *,
    device: str | torch.device,
) -> pd.DataFrame:
    """Return knee-level IoU using optimal one-to-one pairing of two boxes."""
    test_metadata = metadata[metadata["split"].str.lower().eq("test")].copy()
    dataset = DetectorDataset(test_metadata)
    rows = []
    model.eval()
    for index in range(len(dataset)):
        image, target = dataset[index]
        with torch.inference_mode():
            prediction = model([image.to(device)])[0]
        predicted = prediction["boxes"].detach().cpu()[:2]
        reference = target["boxes"]
        ious = box_iou(reference, predicted) if len(predicted) else torch.zeros((2, 0))
        padded_ious = torch.zeros((2, 2), dtype=torch.float32)
        if len(predicted):
            padded_ious[:, : len(predicted)] = ious
        best = None
        for assignment in permutations(range(2), 2):
            values = [float(padded_ious[row, column]) for row, column in enumerate(assignment)]
            if best is None or sum(values) > sum(best):
                best = values
        best = best or [0.0, 0.0]
        participant = test_metadata.iloc[index]["participant_id"]
        for knee_index, value in enumerate(best):
            rows.append(
                {
                    "participant_id": participant,
                    "knee_index": knee_index,
                    "iou": value,
                    "recall_at_0_50": int(value >= 0.50),
                    "recall_at_0_75": int(value >= 0.75),
                }
            )
    return pd.DataFrame(rows)
