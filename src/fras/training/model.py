"""Compact ResNet model architecture for 128x128 face avatar embeddings."""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = cast(torch.Tensor, self.relu(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + res
        return cast(torch.Tensor, self.relu(out))


class AvatarResNet(nn.Module):
    """Compact Residual Network optimized for 128x128 face biometric embedding extraction.

    Outputs unit L2-normalized embeddings of dimension `embedding_dim`.
    """

    def __init__(self, embedding_dim: int = 128, dropout_rate: float = 0.2) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        # Input stem: 128x128x3 -> 64x64x32
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
        )

        # Stage 1: 64x64x32 -> 64x64x32
        self.stage1 = ConvBlock(32, 32, stride=1)
        # Stage 2: 64x64x32 -> 32x32x64
        self.stage2 = ConvBlock(32, 64, stride=2)
        # Stage 3: 32x32x64 -> 16x16x128
        self.stage3 = ConvBlock(64, 128, stride=2)
        # Stage 4: 16x16x128 -> 8x8x256
        self.stage4 = ConvBlock(128, 256, stride=2)

        # Output head: 8x8x256 -> 256 -> embedding_dim
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(256, embedding_dim, bias=False)
        self.bn_head = nn.BatchNorm1d(embedding_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts L2-normalized embeddings from input batch."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn_head(x)
        # Crucial for Cosine/ArcFace metric learning: L2 normalization
        return F.normalize(x, p=2, dim=1)


def create_model(embedding_dim: int = 128, dropout_rate: float = 0.2) -> AvatarResNet:
    return AvatarResNet(embedding_dim=embedding_dim, dropout_rate=dropout_rate)
