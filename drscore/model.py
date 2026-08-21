from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import einops
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class HistogramLayer(nn.Module):
    """Compatibility module retained for released checkpoints; unused in conv mode."""

    def __init__(self, num_bins: int = 20) -> None:
        super().__init__()
        self.numBins = int(num_bins)
        self.bin_centers_conv = nn.Conv2d(1, self.numBins, 1, bias=True)
        self.bin_centers_conv.weight.data.fill_(1)
        self.bin_centers_conv.weight.requires_grad = False
        self.bin_widths_conv = nn.Conv2d(
            self.numBins, self.numBins, 1, groups=self.numBins, bias=False
        )
        self.centers = self.bin_centers_conv.bias
        self.widths = self.bin_widths_conv.weight

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        values = self.bin_widths_conv(self.bin_centers_conv(image))
        values = torch.exp(-(values**2))
        values = values / (values.sum(dim=1, keepdim=True) + 1e-5)
        return F.adaptive_avg_pool2d(values, 1).flatten(1)


class ResNetExtractor(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        rgb_weight = self.model.conv1.weight.detach().clone()
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.conv1.weight.data.copy_(rgb_weight.mean(dim=1, keepdim=True))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


class PatchFeatureExtractor(nn.Module):
    def __init__(self, hist_output_size: int = 20, pretrained: bool = True) -> None:
        super().__init__()
        self.hist_extractor = HistogramLayer(hist_output_size)
        self.conv_extractor = ResNetExtractor(pretrained=pretrained)
        self.cnn_in_size = (224, 224)
        self.emb_size = 1000

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        patches = F.interpolate(patches, self.cnn_in_size, mode="nearest")
        return self.conv_extractor(patches)


class FeatureFC(nn.Module):
    def __init__(self, input_length: int, output_length: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(input_length),
            nn.Linear(input_length, output_length),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


def positional_encoding_1d(
    batch: int, patches: int, dimensions: int, device: torch.device
) -> torch.Tensor:
    if dimensions % 2:
        raise ValueError("The positional-encoding dimension must be even.")
    encoding = torch.zeros(patches, dimensions, device=device)
    position = torch.arange(patches, device=device).unsqueeze(1)
    scale = torch.exp(
        torch.arange(0, dimensions, 2, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / dimensions)
    )
    encoding[:, 0::2] = torch.sin(position.float() * scale)
    encoding[:, 1::2] = torch.cos(position.float() * scale)
    return encoding.repeat(batch, 1, 1)


def spatial_distance_matrix(grid_size: int) -> torch.Tensor:
    coordinates = torch.tensor(
        [(row, column) for row in range(grid_size) for column in range(grid_size)],
        dtype=torch.float32,
    )
    return torch.abs(coordinates[:, None, :] - coordinates[None, :, :]).sum(dim=2)


class KNNSelfAttention(nn.Module):
    def __init__(
        self,
        attention_type: str,
        dimensions: int,
        *,
        representational_neighbors: int | None = None,
        spatial_radius: int | None = None,
        spatial_distances: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.type = attention_type
        self.topk_R = representational_neighbors
        self.topk_S = spatial_radius
        self.spatial_dist_mat = spatial_distances
        self.proj = nn.Linear(dimensions, dimensions, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight.data)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.proj(features)
        similarity = torch.bmm(projected, projected.transpose(1, 2))
        batch, patches, _ = similarity.shape

        if self.type == "representational":
            if self.topk_R is None:
                raise ValueError("representational_neighbors is required.")
            k = min(int(self.topk_R), patches)
            indices = similarity.topk(k=k, dim=-1).indices
            adjacency = torch.eye(patches, device=features.device).repeat(batch, 1, 1)
            adjacency.scatter_(2, indices, 1.0)
        elif self.type == "spatial":
            if self.spatial_dist_mat is None or self.topk_S is None:
                raise ValueError("Spatial distances and spatial_radius are required.")
            adjacency = (
                self.spatial_dist_mat.to(features.device) <= float(self.topk_S)
            ).to(features.dtype).repeat(batch, 1, 1)
        else:
            raise ValueError(f"Unsupported attention type: {self.type}")

        logits = similarity.masked_fill(adjacency == 0, -torch.inf)
        attention = F.softmax(logits, dim=2)
        return torch.bmm(attention, projected), attention


class AttentionRegressor(nn.Module):
    def __init__(self, feature_dimensions: int, attention_dimensions: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dimensions, attention_dimensions),
            nn.Tanh(),
            nn.Linear(attention_dimensions, 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(feature_dimensions, feature_dimensions),
            nn.ReLU(),
            nn.LayerNorm(feature_dimensions),
            nn.Dropout(0.3),
            nn.Linear(feature_dimensions, feature_dimensions),
            nn.ReLU(),
            nn.LayerNorm(feature_dimensions),
            nn.Dropout(0.3),
            nn.Linear(feature_dimensions, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.attention(features).transpose(1, 2)
        self.a = F.softmax(logits, dim=2)
        pooled = torch.bmm(self.a, features).squeeze(1)
        return self.fc(pooled).flatten().float()


class _DRScoreCore(nn.Module):
    def __init__(
        self,
        grid_size: int,
        patch_size: int,
        feature_embedding_size: int,
        attention_embedding_size: int,
        representational_neighbors: int,
        spatial_radius: int,
        hist_output_size: int,
        pretrained: bool,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.patch_size = int(patch_size)
        self.feat_extractor = PatchFeatureExtractor(hist_output_size, pretrained)
        self.embedding_fc = FeatureFC(1000, feature_embedding_size)
        self.RkNNAtt = KNNSelfAttention(
            "representational",
            feature_embedding_size,
            representational_neighbors=representational_neighbors,
        )
        self.SkNNAtt = KNNSelfAttention(
            "spatial",
            feature_embedding_size,
            spatial_radius=spatial_radius,
            spatial_distances=spatial_distance_matrix(self.grid_size),
        )
        self.attention_regressor = AttentionRegressor(
            feature_embedding_size, attention_embedding_size
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = image.shape
        expected = self.grid_size * self.patch_size
        if channels != 1 or height != expected or width != expected:
            raise ValueError(f"Expected Bx1x{expected}x{expected}, got {tuple(image.shape)}")
        patches = einops.rearrange(
            image,
            "b c (rows ph) (cols pw) -> (b c rows cols) 1 ph pw",
            rows=self.grid_size,
            cols=self.grid_size,
            ph=self.patch_size,
            pw=self.patch_size,
        )
        features = self.feat_extractor(patches)
        features = einops.rearrange(
            features,
            "(b patches) features -> b patches features",
            b=batch,
            patches=self.grid_size**2,
        )
        features = self.embedding_fc(features)
        features = features + positional_encoding_1d(
            batch, self.grid_size**2, features.shape[-1], features.device
        )
        represented, self.rknn_att = self.RkNNAtt(features)
        spatial, self.sknn_att = self.SkNNAtt(features)
        risk = self.attention_regressor((represented + spatial) / 2.0)
        self.a = self.attention_regressor.a
        return risk


class DRScoreNetwork(nn.Module):
    """Compatibility wrapper whose state-dict names match the trained model."""

    def __init__(self, **model_config: Any) -> None:
        super().__init__()
        self.model = _DRScoreCore(**model_config)

    @property
    def attention(self) -> torch.Tensor | None:
        return getattr(self.model, "a", None)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


def build_drscore_model(config: dict[str, Any], *, pretrained: bool = True) -> DRScoreNetwork:
    return DRScoreNetwork(
        grid_size=int(config["grid_size"]),
        patch_size=int(config["patch_size"]),
        feature_embedding_size=int(config["feature_embedding_size"]),
        attention_embedding_size=int(config["attention_embedding_size"]),
        representational_neighbors=int(config["representational_neighbors"]),
        spatial_radius=int(config["spatial_radius"]),
        hist_output_size=int(config.get("hist_output_size", 20)),
        pretrained=pretrained,
    )


def load_drscore_checkpoint(
    path: str | Path,
    config: dict[str, Any],
    device: str | torch.device = "cpu",
) -> tuple[DRScoreNetwork, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    model = build_drscore_model(config, pretrained=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match the configured model; missing={missing}, "
            f"unexpected={unexpected}"
        )
    model.to(device).eval()
    return model, checkpoint.get("metadata", {})
