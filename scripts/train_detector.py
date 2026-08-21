from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from drscore.detector import evaluate_detector, load_detector_checkpoint, train_detector


def load_detector_metadata(path: Path) -> pd.DataFrame:
    path = path.expanduser().resolve()
    frame = pd.read_csv(path)
    for column in ("image_path", "annotation_path"):
        frame[column] = frame[column].map(
            lambda value: str(
                (path.parent / value).resolve()
                if not Path(value).is_absolute()
                else Path(value).resolve()
            )
        )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate the knee detector.")
    parser.add_argument("metadata_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/knee_detector.pt"))
    parser.add_argument("--allow-other-counts", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    metadata = load_detector_metadata(args.metadata_csv)
    counts = metadata["split"].str.lower().value_counts().to_dict()
    if not args.allow_other_counts and (counts.get("train") != 100 or counts.get("test") != 61):
        raise ValueError(
            "Manuscript reproduction expects 100 training and 61 independent test radiographs. "
            "Use --allow-other-counts only for a separate dataset."
        )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = train_detector(
        metadata,
        args.output,
        device=device,
        num_workers=args.num_workers,
    )
    detector, _ = load_detector_checkpoint(checkpoint, device=device)
    evaluation = evaluate_detector(detector, metadata, device=device)
    output_csv = checkpoint.with_name("knee_detector_test_iou.csv")
    evaluation.to_csv(output_csv, index=False)
    print(
        evaluation[["iou", "recall_at_0_50", "recall_at_0_75"]]
        .mean()
        .to_string()
    )
    print(f"Knee-level results: {output_csv}")


if __name__ == "__main__":
    main()
