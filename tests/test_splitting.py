import pandas as pd

from drscore.splitting import participant_grouped_splits


def test_participant_splits_do_not_leak():
    rows = []
    for participant in range(80):
        for side in ("left", "right"):
            rows.append(
                {
                    "participant_id": f"P{participant:03d}",
                    "event": int(participant % 4 == 0),
                    "side": side,
                }
            )
    frame = pd.DataFrame(rows)
    for split in participant_grouped_splits(frame, outer_folds=5, inner_folds=8, seed=1029):
        train = set(frame.loc[split.train_index, "participant_id"])
        validation = set(frame.loc[split.validation_index, "participant_id"])
        test = set(frame.loc[split.test_index, "participant_id"])
        assert not train & validation
        assert not train & test
        assert not validation & test
