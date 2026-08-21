from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import ImageDraw, ImageFont, ImageOps

from .config import load_config
from .detector import detect_and_crop_bilateral, load_detector_checkpoint
from .model import load_drscore_checkpoint
from .preprocessing import DRScorePreprocessor, load_image, rescale_dr_score


def _resolve(base: Path, path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def infer_bilateral_image(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path = "configs/final_model.yaml",
    patient_right_on_image_left: bool | None = None,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    config_path = Path(config_path).expanduser().resolve()
    repository = config_path.parent.parent
    config = load_config(config_path)
    inference = config["inference"]
    preprocessing = config["preprocessing"]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    detector, _ = load_detector_checkpoint(
        _resolve(repository, inference["detector_checkpoint"]), device=device
    )
    risk_model, _ = load_drscore_checkpoint(
        _resolve(repository, inference["drscore_checkpoint"]),
        config["model"],
        device=device,
    )
    image = load_image(image_path)
    image = ImageOps.exif_transpose(image).convert("L")
    right_on_left = (
        bool(inference["patient_right_on_image_left"])
        if patient_right_on_image_left is None
        else bool(patient_right_on_image_left)
    )
    detections = detect_and_crop_bilateral(
        detector,
        image,
        device=device,
        score_threshold=float(inference["detector_score_threshold"]),
        nms_threshold=float(inference["detector_nms_threshold"]),
        margin_pixels=int(inference["crop_margin_pixels"]),
        patient_right_on_image_left=right_on_left,
    )
    processor = DRScorePreprocessor(
        image_size=int(preprocessing["image_size"]),
        clahe_clip_limit=float(preprocessing["clahe_clip_limit"]),
        clahe_tile_grid=tuple(preprocessing["clahe_tile_grid"]),
    )
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=max(18, overlay.width // 70))
    colors = {"left": "#00A878", "right": "#D1495B"}
    rows = []
    for side in ("left", "right"):
        record = detections[side]
        crop = record["crop"].convert("L")
        crop.save(output_dir / f"{side}_knee_crop.png")
        model_orientation = ImageOps.mirror(crop) if side == "right" else crop
        model_orientation.save(output_dir / f"{side}_knee_model_orientation.png")
        tensor = processor(crop, side=side, reflect_right_knee=True).unsqueeze(0).to(device)
        with torch.inference_mode():
            raw = float(risk_model(tensor).item())
        score = rescale_dr_score(
            raw,
            float(inference["raw_score_min"]),
            float(inference["raw_score_max"]),
            clip=bool(inference["clip_to_0_4"]),
        )
        x1, y1, x2, y2 = record["crop_box"]
        draw.rectangle((x1, y1, x2, y2), outline=colors[side], width=5)
        label = f"{side}: {score:.3f}"
        label_box = draw.textbbox((x1 + 6, y1 + 6), label, font=font)
        draw.rectangle(
            (label_box[0] - 4, label_box[1] - 3, label_box[2] + 4, label_box[3] + 3),
            fill=colors[side],
        )
        draw.text((x1 + 6, y1 + 6), label, fill="white", font=font)
        rows.append(
            {
                "side": side,
                "dr_score": score,
                "raw_prediction": raw,
                "detector_confidence": record["confidence"],
                "crop_xmin": x1,
                "crop_ymin": y1,
                "crop_xmax": x2,
                "crop_ymax": y2,
            }
        )
    overlay.save(output_dir / "bilateral_detection_and_scores.png")
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "dr_scores.csv", index=False)
    with (output_dir / "inference_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_filename": Path(image_path).name,
                "patient_right_on_image_left": right_on_left,
                "score_scaling": {
                    "raw_min": inference["raw_score_min"],
                    "raw_max": inference["raw_score_max"],
                    "clip_to_0_4": inference["clip_to_0_4"],
                },
            },
            handle,
            indent=2,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate left and right DR Scores from one bilateral knee radiograph."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/single_inference"))
    parser.add_argument("--config", type=Path, default=Path("configs/final_model.yaml"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument(
        "--patient-left-on-image-left",
        action="store_true",
        help="Use when the image is not displayed in the standard radiographic convention.",
    )
    args = parser.parse_args()
    result = infer_bilateral_image(
        args.image,
        args.output_dir,
        config_path=args.config,
        patient_right_on_image_left=not args.patient_left_on_image_left,
        device=args.device,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
