from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import DRScorePreprocessor, load_image


REQUIRED_COLUMNS = {
    "cohort",
    "participant_id",
    "knee_id",
    "side",
    "image_path",
    "time_months",
    "event",
}


def load_metadata(
    path: str | Path,
    *,
    allowed_cohorts: tuple[str, ...] = ("OAI", "MOST", "KICK", "MenTOR"),
) -> pd.DataFrame:
    metadata_path = Path(path).expanduser().resolve()
    frame = pd.read_csv(metadata_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    frame = frame[list(REQUIRED_COLUMNS)].copy()
    if frame.isna().any().any():
        columns = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"Metadata contains missing values in: {columns}")

    frame["cohort"] = frame["cohort"].astype(str)
    unknown = sorted(set(frame["cohort"]) - set(allowed_cohorts))
    if unknown:
        raise ValueError(f"Unknown cohort label(s): {unknown}")
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["knee_id"] = frame["knee_id"].astype(str)
    frame["side"] = frame["side"].astype(str).str.lower()
    if not frame["side"].isin(["left", "right"]).all():
        raise ValueError("side must contain only 'left' or 'right'.")
    frame["time_months"] = pd.to_numeric(frame["time_months"], errors="raise")
    frame["event"] = pd.to_numeric(frame["event"], errors="raise").astype(int)
    if not frame["event"].isin([0, 1]).all():
        raise ValueError("event must contain only 0 or 1.")
    if (frame["time_months"] <= 0).any():
        raise ValueError("time_months must be positive.")
    if frame["knee_id"].duplicated().any():
        raise ValueError("knee_id values must be unique.")

    def resolve_image(value: str) -> str:
        image_path = Path(value).expanduser()
        if not image_path.is_absolute():
            image_path = metadata_path.parent / image_path
        return str(image_path.resolve())

    frame["image_path"] = frame["image_path"].astype(str).map(resolve_image)
    absent = [path for path in frame["image_path"] if not Path(path).is_file()]
    if absent:
        raise FileNotFoundError(f"Image not found: {absent[0]}")
    return frame.reset_index(drop=True)


class KneeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, preprocessor: DRScorePreprocessor) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image = load_image(row["image_path"])
        tensor = self.preprocessor(image, side=row["side"], reflect_right_knee=True)
        event = torch.tensor(float(row["event"]), dtype=torch.float32)
        time = torch.tensor(float(row["time_months"]), dtype=torch.float32)
        return tensor, event, time


def make_loader(
    frame: pd.DataFrame,
    preprocessor: DRScorePreprocessor,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        KneeDataset(frame, preprocessor),
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
