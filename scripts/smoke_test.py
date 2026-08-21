from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from drscore.config import load_config
from drscore.detector import load_detector_checkpoint, square_crop_box
from drscore.model import load_drscore_checkpoint
from drscore.preprocessing import DRScorePreprocessor, rescale_dr_score
from drscore.splitting import participant_grouped_splits


def main() -> None:
    root = ROOT
    config = load_config(root / "configs" / "final_model.yaml")

    processed = DRScorePreprocessor(image_size=70)(
        np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    )
    assert processed.shape == (1, 70, 70)
    assert abs(float(processed.mean())) < 0.02
    assert rescale_dr_score(-4.236161, -4.236161, 3.5887303) == 0.0
    assert square_crop_box(torch.tensor([100, 80, 300, 260]), (500, 400), margin_pixels=10) == (
        90, 60, 310, 280
    )

    frame = pd.DataFrame(
        [
            {"participant_id": f"P{participant:03d}", "event": int(participant % 4 == 0)}
            for participant in range(80)
            for _ in range(2)
        ]
    )
    for split in participant_grouped_splits(frame, seed=1029):
        train = set(frame.loc[split.train_index, "participant_id"])
        validation = set(frame.loc[split.validation_index, "participant_id"])
        test = set(frame.loc[split.test_index, "participant_id"])
        assert not train & validation and not train & test and not validation & test

    risk_model, _ = load_drscore_checkpoint(
        root / config["inference"]["drscore_checkpoint"], config["model"]
    )
    detector, _ = load_detector_checkpoint(root / config["inference"]["detector_checkpoint"])
    assert sum(parameter.numel() for parameter in risk_model.parameters()) > 10_000_000
    assert sum(parameter.numel() for parameter in detector.parameters()) > 40_000_000

    subprocess.run([sys.executable, str(root / "scripts" / "privacy_check.py")], check=True)
    print("Smoke test passed: configuration, preprocessing, geometry, splits, and checkpoints are valid.")


if __name__ == "__main__":
    main()
