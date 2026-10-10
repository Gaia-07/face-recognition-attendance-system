"""Temporal multi-frame voting engine for robust sequential video consensus."""

from collections import Counter, deque
from dataclasses import dataclass


@dataclass
class VotingDecision:
    confirmed_id: str | None
    is_confirmed: bool
    votes: int
    window_size: int
    consensus_ratio: float


class TemporalVotingEngine:
    """Multi-frame voting consensus engine.

    Requires $k$ consistent identity predictions within the last $n$ frames
    before confirming biometric check-in.
    """

    def __init__(self, k: int = 3, n: int = 5) -> None:
        self.k = k
        self.n = n
        self.history: deque[str | None] = deque(maxlen=n)

    def reset(self) -> None:
        self.history.clear()

    def update(self, predicted_id: str | None) -> VotingDecision:
        """Appends latest frame prediction and computes temporal consensus."""
        self.history.append(predicted_id)

        # Count occurrences of non-None identities in current window
        valid_predictions = [x for x in self.history if x is not None]
        if not valid_predictions:
            return VotingDecision(
                confirmed_id=None,
                is_confirmed=False,
                votes=0,
                window_size=len(self.history),
                consensus_ratio=0.0,
            )

        counts = Counter(valid_predictions)
        mode_id, count = counts.most_common(1)[0]
        consensus_ratio = count / float(self.n)

        if count >= self.k:
            return VotingDecision(
                confirmed_id=mode_id,
                is_confirmed=True,
                votes=count,
                window_size=len(self.history),
                consensus_ratio=consensus_ratio,
            )

        return VotingDecision(
            confirmed_id=None,
            is_confirmed=False,
            votes=count,
            window_size=len(self.history),
            consensus_ratio=consensus_ratio,
        )
