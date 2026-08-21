from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class ParticipantSplit:
    outer_fold: int
    train_index: np.ndarray
    validation_index: np.ndarray
    test_index: np.ndarray


def participant_grouped_splits(
    frame: pd.DataFrame,
    *,
    outer_folds: int = 5,
    inner_folds: int = 8,
    seed: int = 1029,
) -> list[ParticipantSplit]:
    """Create approximately 70/10/20 participant-disjoint partitions."""
    participant = (
        frame.groupby("participant_id", as_index=False)
        .agg(event=("event", "max"))
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    splits: list[ParticipantSplit] = []
    for outer_fold, (development, test) in enumerate(
        outer.split(participant["participant_id"], participant["event"]), start=1
    ):
        development_participants = participant.iloc[development].reset_index(drop=True)
        inner = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed + outer_fold,
        )
        train_local, validation_local = next(
            inner.split(
                development_participants["participant_id"],
                development_participants["event"],
            )
        )
        train_ids = set(development_participants.iloc[train_local]["participant_id"])
        validation_ids = set(
            development_participants.iloc[validation_local]["participant_id"]
        )
        test_ids = set(participant.iloc[test]["participant_id"])
        if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
            raise RuntimeError("Participant leakage was detected.")
        splits.append(
            ParticipantSplit(
                outer_fold=outer_fold,
                train_index=frame.index[frame["participant_id"].isin(train_ids)].to_numpy(),
                validation_index=frame.index[
                    frame["participant_id"].isin(validation_ids)
                ].to_numpy(),
                test_index=frame.index[frame["participant_id"].isin(test_ids)].to_numpy(),
            )
        )
    return splits


def split_manifest(frame: pd.DataFrame, splits: list[ParticipantSplit]) -> pd.DataFrame:
    rows = []
    for split in splits:
        for name, indices in (
            ("train", split.train_index),
            ("validation", split.validation_index),
            ("test", split.test_index),
        ):
            subset = frame.loc[indices]
            rows.append(
                {
                    "outer_fold": split.outer_fold,
                    "partition": name,
                    "n_knees": len(subset),
                    "n_participants": subset["participant_id"].nunique(),
                    "events": int(subset["event"].sum()),
                    "event_proportion": float(subset["event"].mean()),
                }
            )
    return pd.DataFrame(rows)
