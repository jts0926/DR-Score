from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from .data import make_loader
from .model import DRScoreNetwork, build_drscore_model
from .preprocessing import DRScorePreprocessor, rescale_dr_score
from .splitting import participant_grouped_splits, split_manifest
from .survival import cox_negative_partial_log_likelihood


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainingResult:
    best_epoch: int
    best_validation_loss: float
    checkpoint_path: Path
    history: pd.DataFrame


def _run_epoch(
    model: DRScoreNetwork,
    loader,
    *,
    device: torch.device,
    l2_coefficient: float,
    optimizer: AdamW | None,
    accumulation_steps: int = 1,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, (image, event, time) in enumerate(loader, start=1):
        image, event, time = image.to(device), event.to(device), time.to(device)
        with torch.set_grad_enabled(training):
            risk = model(image)
            loss = cox_negative_partial_log_likelihood(
                risk,
                time,
                event,
                model=model,
                l2_coefficient=l2_coefficient,
            )
            if training:
                (loss / accumulation_steps).backward()
                if batch_index % accumulation_steps == 0 or batch_index == len(loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu())
        batches += 1
    return total_loss / max(batches, 1)


def predict_frame(
    model: DRScoreNetwork,
    frame: pd.DataFrame,
    preprocessor: DRScorePreprocessor,
    *,
    batch_size: int,
    device: torch.device,
    raw_min: float,
    raw_max: float,
    clip: bool,
    num_workers: int = 0,
) -> pd.DataFrame:
    loader = make_loader(
        frame,
        preprocessor,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    predictions = []
    model.eval()
    with torch.inference_mode():
        for image, _, _ in loader:
            predictions.extend(model(image.to(device)).cpu().numpy().tolist())
    output = frame.reset_index(drop=True).copy()
    output["raw_prediction"] = np.asarray(predictions, dtype=float)
    output["dr_score"] = rescale_dr_score(
        output["raw_prediction"].to_numpy(), raw_min, raw_max, clip=clip
    )
    return output


def train_fold(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    config: dict[str, Any],
    output_dir: Path,
    fold_name: str,
    device: torch.device,
    seed: int,
    num_workers: int = 0,
) -> tuple[DRScoreNetwork, TrainingResult]:
    model_config = config["model"]
    training = config["training"]
    preprocessing = config["preprocessing"]
    seed_everything(seed)
    model = build_drscore_model(model_config, pretrained=True).to(device)
    processor = DRScorePreprocessor(
        image_size=preprocessing["image_size"],
        clahe_clip_limit=preprocessing["clahe_clip_limit"],
        clahe_tile_grid=tuple(preprocessing["clahe_tile_grid"]),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = make_loader(
        train_frame,
        processor,
        batch_size=training["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    validation_loader = make_loader(
        validation_frame,
        processor,
        batch_size=training["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        amsgrad=bool(training["amsgrad"]),
    )
    scheduler = ExponentialLR(optimizer, gamma=float(training["scheduler_gamma"]))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"drscore_{fold_name}.pt"
    log_dir = output_dir / "tensorboard" / fold_name
    writer = SummaryWriter(log_dir=str(log_dir))

    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, int(training["max_epochs"]) + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            device=device,
            l2_coefficient=float(training["cox_l2_coefficient"]),
            optimizer=optimizer,
            accumulation_steps=int(training["gradient_accumulation_steps"]),
        )
        validation_loss = _run_epoch(
            model,
            validation_loader,
            device=device,
            l2_coefficient=float(training["cox_l2_coefficient"]),
            optimizer=None,
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalars(
            "loss", {"train": train_loss, "validation": validation_loss}, epoch
        )
        writer.add_scalar("learning_rate", current_lr, epoch)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": current_lr,
            }
        )
        print(
            f"{fold_name} epoch {epoch:02d}: train={train_loss:.5f}, "
            f"validation={validation_loss:.5f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "metadata": {
                        "format_version": 1,
                        "fold": fold_name,
                        "model": model_config,
                        "preprocessing": preprocessing,
                    },
                },
                checkpoint_path,
            )
        elif epoch >= int(training["minimum_epochs"]):
            epochs_without_improvement += 1
        if (
            epoch >= int(training["minimum_epochs"])
            and epochs_without_improvement >= int(training["early_stopping_patience"])
        ):
            break
    writer.close()
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    result = TrainingResult(
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        checkpoint_path=checkpoint_path,
        history=pd.DataFrame(history),
    )
    return model, result


def run_cross_validation(
    frame: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    seeds: list[int] | None = None,
    device: str | torch.device | None = None,
    num_workers: int = 0,
) -> pd.DataFrame:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    inference = config["inference"]
    preprocessing = config["preprocessing"]
    seeds = seeds or [int(training["primary_seed"])]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processor = DRScorePreprocessor(
        image_size=preprocessing["image_size"],
        clahe_clip_limit=preprocessing["clahe_clip_limit"],
        clahe_tile_grid=tuple(preprocessing["clahe_tile_grid"]),
    )
    selection_rows = []
    for seed in seeds:
        seed_everything(seed)
        splits = participant_grouped_splits(
            frame,
            outer_folds=int(training["outer_folds"]),
            inner_folds=round(1 / float(training["validation_fraction_of_development"])),
            seed=seed,
        )
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        split_manifest(frame, splits).to_csv(seed_dir / "split_manifest.csv", index=False)
        for split in splits:
            fold_name = f"outer{split.outer_fold}"
            fold_dir = seed_dir / fold_name
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_frame = frame.loc[split.train_index].copy()
            validation_frame = frame.loc[split.validation_index].copy()
            test_frame = frame.loc[split.test_index].copy()
            model, result = train_fold(
                train_frame,
                validation_frame,
                config=config,
                output_dir=fold_dir,
                fold_name=fold_name,
                device=device,
                seed=seed,
                num_workers=num_workers,
            )
            result.history.to_csv(fold_dir / "training_history.csv", index=False)
            for partition, partition_frame in (
                ("train", train_frame),
                ("validation", validation_frame),
                ("test", test_frame),
            ):
                predictions = predict_frame(
                    model,
                    partition_frame,
                    processor,
                    batch_size=int(training["batch_size"]),
                    device=device,
                    raw_min=float(inference["raw_score_min"]),
                    raw_max=float(inference["raw_score_max"]),
                    clip=bool(inference["clip_to_0_4"]),
                    num_workers=num_workers,
                )
                predictions.to_csv(fold_dir / f"{partition}_dr_scores.csv", index=False)
            selection_rows.append(
                {
                    "seed": seed,
                    "outer_fold": split.outer_fold,
                    "best_epoch": result.best_epoch,
                    "best_validation_loss": result.best_validation_loss,
                    "checkpoint": str(result.checkpoint_path.relative_to(output_dir)),
                }
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    selection = pd.DataFrame(selection_rows).sort_values(
        ["seed", "best_validation_loss", "outer_fold"]
    )
    selection["selected_within_seed"] = selection.groupby("seed").cumcount().eq(0)
    selection.to_csv(output_dir / "model_selection.csv", index=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return selection
