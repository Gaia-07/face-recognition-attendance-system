"""Additive Angular Margin (ArcFace) loss head for metric learning."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """ArcFace: Additive Angular Margin Loss head.

    Reference: Deng et al. 'ArcFace: Additive Angular Margin Loss for Deep Face Recognition', CVPR 2019.

    Args:
        in_features: Size of input embedding vector (e.g., 128).
        num_classes: Number of training identity classes.
        scale: Feature scale parameter (s), typically 30.0 - 64.0.
        margin: Additive angular margin parameter (m) in radians, typically 0.3 - 0.5.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 32.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Weight matrix: (num_classes, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass computing scaled angular margin logits.

        Args:
            embeddings: L2-normalized feature embeddings (N, in_features).
            labels: Ground truth identity class indices (N).

        Returns:
            Logits of shape (N, num_classes) ready for standard CrossEntropyLoss.
        """
        # Normalize weights to unit hypersphere
        w_norm = F.normalize(self.weight, p=2, dim=1)
        # Cosine similarity: cos(theta) = x * w.T
        cosine = F.linear(embeddings, w_norm)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # sin(theta) = sqrt(1 - cos(theta)^2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot mask for ground truth labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Apply angular margin exclusively to ground truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale
        return output
