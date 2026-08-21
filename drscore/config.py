from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the public YAML configuration."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"data", "preprocessing", "model", "training", "inference"}
    missing = required.difference(config or {})
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")

    model = config["model"]
    image_size = int(config["preprocessing"]["image_size"])
    grid_size = int(model["grid_size"])
    patch_size = int(model["patch_size"])
    if image_size != grid_size * patch_size:
        raise ValueError("image_size must equal grid_size * patch_size.")
    if int(model["representational_neighbors"]) < 1:
        raise ValueError("representational_neighbors must be positive.")
    if int(model["spatial_radius"]) < 0:
        raise ValueError("spatial_radius cannot be negative.")
    return config
