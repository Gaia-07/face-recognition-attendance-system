"""Biometric vector matcher with calibrated thresholding, margin gating, and open-set rejection."""

from dataclasses import dataclass

import numpy as np


@dataclass
class MatchCandidate:
    identity_id: str
    similarity: float


@dataclass
class MatchResult:
    matched_id: str | None
    similarity: float
    margin: float
    is_match: bool
    status: str  # "MATCHED", "REJECT_UNKNOWN", "REJECT_AMBIGUOUS_MARGIN", "NO_ENROLLED_TEMPLATES"
    candidates: list[MatchCandidate]


class BiometricMatcher:
    """Gallery vector matcher computing normalized cosine similarities."""

    def __init__(
        self,
        threshold: float = 0.653,
        margin_threshold: float = 0.05,
        top_k: int = 3,
    ) -> None:
        self.threshold = threshold
        self.margin_threshold = margin_threshold
        self.top_k = top_k
        self.templates: dict[str, np.ndarray] = {}  # id -> averaged unit-vector template

    def enroll(self, identity_id: str, embedding: np.ndarray) -> None:
        """Enrolls or updates an identity embedding template."""
        emb = np.asarray(embedding, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(emb))
        unit_emb = emb / (norm + 1e-7)

        if identity_id in self.templates:
            # Running average update
            current = self.templates[identity_id]
            updated = (current + unit_emb) / 2.0
            updated_norm = float(np.linalg.norm(updated))
            self.templates[identity_id] = updated / (updated_norm + 1e-7)
        else:
            self.templates[identity_id] = unit_emb

    def clear(self) -> None:
        self.templates.clear()

    def remove(self, identity_id: str) -> bool:
        if identity_id in self.templates:
            del self.templates[identity_id]
            return True
        return False

    def match(self, probe_embedding: np.ndarray) -> MatchResult:
        """Matches a probe embedding against all enrolled templates.

        Returns MatchResult with open-set rejection and margin verification.
        """
        if not self.templates:
            return MatchResult(
                matched_id=None,
                similarity=0.0,
                margin=0.0,
                is_match=False,
                status="NO_ENROLLED_TEMPLATES",
                candidates=[],
            )

        probe = np.asarray(probe_embedding, dtype=np.float32).flatten()
        p_norm = float(np.linalg.norm(probe))
        unit_probe = probe / (p_norm + 1e-7)

        identities = list(self.templates.keys())
        matrix = np.stack([self.templates[pid] for pid in identities])  # (M, D)

        # Cosine similarity matrix multiplication
        sims = np.dot(matrix, unit_probe)  # (M,)
        ranked_indices = np.argsort(-sims)

        k = min(self.top_k, len(identities))
        candidates: list[MatchCandidate] = [
            MatchCandidate(identity_id=identities[idx], similarity=float(sims[idx]))
            for idx in ranked_indices[:k]
        ]

        best_idx = ranked_indices[0]
        best_sim = float(sims[best_idx])
        best_id = identities[best_idx]

        # Margin calculation: difference between 1st and 2nd best matches
        if len(ranked_indices) > 1:
            second_sim = float(sims[ranked_indices[1]])
            margin = best_sim - second_sim
        else:
            margin = 1.0  # Only one template enrolled

        # Open-set threshold check
        if best_sim < self.threshold:
            return MatchResult(
                matched_id=None,
                similarity=best_sim,
                margin=margin,
                is_match=False,
                status="REJECT_UNKNOWN",
                candidates=candidates,
            )

        # Ambiguity margin check
        if len(ranked_indices) > 1 and margin < self.margin_threshold:
            return MatchResult(
                matched_id=None,
                similarity=best_sim,
                margin=margin,
                is_match=False,
                status="REJECT_AMBIGUOUS_MARGIN",
                candidates=candidates,
            )

        return MatchResult(
            matched_id=best_id,
            similarity=best_sim,
            margin=margin,
            is_match=True,
            status="MATCHED",
            candidates=candidates,
        )
