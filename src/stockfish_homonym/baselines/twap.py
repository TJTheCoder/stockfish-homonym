from __future__ import annotations

import numpy as np
import torch


_IDX_TIME_REMAINING = 0
_IDX_INVENTORY = 1
_IDX_URGENCY = 17


def platform_twap_action(obs: np.ndarray) -> int:
    """Return the built-in TWAP action for a single environment observation."""
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


def platform_twap_actions_torch(obs: torch.Tensor) -> torch.Tensor:
    """Vectorized TWAP actions for batched observation tensors with shape (..., obs_dim)."""
    time_remaining_frac = obs[..., _IDX_TIME_REMAINING]
    inventory_frac = obs[..., _IDX_INVENTORY]
    urgency = obs[..., _IDX_URGENCY] * 5.0

    elapsed_frac = 1.0 - time_remaining_frac
    filled_frac = 1.0 - inventory_frac
    ahead_of_schedule = filled_frac >= elapsed_frac

    actions = torch.ones_like(time_remaining_frac, dtype=torch.long)
    actions = torch.where(urgency > 1.0, torch.full_like(actions, 2), actions)
    actions = torch.where(urgency > 1.4, torch.full_like(actions, 3), actions)
    actions = torch.where(urgency > 2.0, torch.full_like(actions, 4), actions)
    actions = torch.where(ahead_of_schedule, torch.zeros_like(actions), actions)
    actions = torch.where(
        time_remaining_frac < 0.1, torch.full_like(actions, 5), actions
    )
    return actions


class PlatformTwapAgent:
    """Deterministic schedule baseline for the C++ execution task."""

    def __init__(self, target_inventory: int, horizon: int) -> None:
        self._target_inventory = target_inventory
        self._horizon = horizon

    def act(self, obs: np.ndarray) -> int:
        return platform_twap_action(obs)
