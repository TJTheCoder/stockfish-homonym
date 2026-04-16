from __future__ import annotations

import numpy as np


_IDX_TIME_REMAINING = 0
_IDX_INVENTORY = 1
_IDX_URGENCY = 17


class PlatformTwapAgent:
    """Deterministic schedule baseline for the C++ execution task."""

    def __init__(self, target_inventory: int, horizon: int) -> None:
        self._target_inventory = target_inventory
        self._horizon = horizon

    def act(self, obs: np.ndarray) -> int:
        time_remaining_frac = float(obs[_IDX_TIME_REMAINING])
        inventory_frac = float(obs[_IDX_INVENTORY])
        urgency = float(obs[_IDX_URGENCY]) * 5.0

        elapsed_frac = 1.0 - time_remaining_frac
        target_filled_frac = elapsed_frac
        filled_frac = 1.0 - inventory_frac

        if time_remaining_frac < 0.1:
            return 5
        if filled_frac >= target_filled_frac:
            return 0
        if urgency > 2.0:
            return 4
        if urgency > 1.4:
            return 3
        if urgency > 1.0:
            return 2
        return 1
