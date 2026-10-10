"""EmbeddingBackend protocol and registry for modular biometric feature extractors."""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol defining the interface for all face embedding backends.

    All backends must output unit L2-normalized feature vectors of shape (N, dim).
    """

    name: str
    dim: int

    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """Transforms input RGB image batch (N, H, W, 3) to backend-specific input format."""
        ...

    def embed(self, batch: np.ndarray) -> np.ndarray:
        """Extracts unit L2-normalized feature embeddings.

        Args:
            batch: Preprocessed image tensor batch (N, C, H, W) or (N, H, W, C).

        Returns:
            Embeddings of shape (N, dim) satisfying norm(emb) == 1.0.
        """
        ...
