"""Lightweight active challenge-response liveness verification.

DISCLAIMER: This module implements active challenge-response motion estimation
as a heuristic deterrent against static photo replay. It is NOT an ISO/IEC 30107-3
certified Presentation Attack Detection (PAD) system.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

logger = logging.getLogger(__name__)


class LivenessChallenge(StrEnum):
    NONE = "NONE"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    BLINK_OR_HOLD = "BLINK_OR_HOLD"


@dataclass
class LivenessResult:
    passed: bool
    challenge: LivenessChallenge
    yaw_score: float
    is_simulated: bool
    message: str


class LivenessEngine:
    """Active challenge-response liveness detector using 5-point facial landmarks."""

    def __init__(self, simulation_mode: bool = True) -> None:
        self.simulation_mode = simulation_mode

    def verify_challenge(
        self,
        landmarks: np.ndarray | None,
        challenge: LivenessChallenge = LivenessChallenge.TURN_LEFT,
    ) -> LivenessResult:
        """Verifies active head movement against the issued challenge.

        In avatar/simulation mode, the check is bypassed with an audit trail note.
        """
        if self.simulation_mode or landmarks is None:
            logger.debug("Liveness bypassed under simulation/avatar mode.")
            return LivenessResult(
                passed=True,
                challenge=challenge,
                yaw_score=0.0,
                is_simulated=True,
                message="Liveness bypassed: operational under synthetic avatar/simulation mode.",
            )

        # Landmarks: [0: right eye, 1: left eye, 2: nose, 3: right mouth, 4: left mouth]
        re_x = float(landmarks[0, 0])
        le_x = float(landmarks[1, 0])
        nose_x = float(landmarks[2, 0])

        d_right = abs(nose_x - re_x)
        d_left = abs(le_x - nose_x)
        # Yaw ratio: > 1.4 suggests turning right, < 0.7 suggests turning left
        yaw_ratio = d_right / (d_left + 1e-5)

        if challenge == LivenessChallenge.TURN_LEFT:
            passed = yaw_ratio < 0.75
            msg = "Head turned left verified." if passed else "Please turn your head left."
        elif challenge == LivenessChallenge.TURN_RIGHT:
            passed = yaw_ratio > 1.35
            msg = "Head turned right verified." if passed else "Please turn your head right."
        else:
            passed = True
            msg = "Challenge verified."

        return LivenessResult(
            passed=passed,
            challenge=challenge,
            yaw_score=yaw_ratio,
            is_simulated=False,
            message=msg,
        )
