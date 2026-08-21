from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drscore.config import load_config
from drscore.data import load_metadata
from drscore.training import run_cross_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the participant-grouped DR Score models.")
    parser.add_argument("--config", type=Path, default=Path("configs/final_model.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cross_validation"))
    parser.add_argument("--include-sensitivity-seeds", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    repository = args.config.resolve().parent.parent
    metadata_path = Path(config["data"]["metadata_csv"])
    if not metadata_path.is_absolute():
        metadata_path = repository / metadata_path
    frame = load_metadata(
        metadata_path,
        allowed_cohorts=tuple(config["data"]["allowed_cohorts"]),
    )
    seeds = [int(config["training"]["primary_seed"])]
    if args.include_sensitivity_seeds:
        seeds.extend(map(int, config["training"]["sensitivity_seeds"]))
    selection = run_cross_validation(
        frame,
        config,
        args.output_dir,
        seeds=seeds,
        device=args.device,
        num_workers=args.num_workers,
    )
    print(selection.to_string(index=False))


if __name__ == "__main__":
    main()
