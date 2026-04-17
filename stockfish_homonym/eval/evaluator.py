from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class Agent(Protocol):
    def act(self, obs: np.ndarray) -> int: ...


@dataclass
class EvalMetrics:
    shortfall_mean: float
    shortfall_std: float
    fill_rate: float
    regime_shortfall: dict[str, float]
    action_dist: dict[int, float]


class Evaluator:
    _REGIME_NAMES = {
        0: "calm",
        1: "normal",
        2: "stressed",
    }

    def __init__(
        self,
        env_factory: Callable[[int], object],
        n_episodes: int = 100,
        seed_offset: int = 10_000,
    ) -> None:
        self._env_factory = env_factory
        self._n_episodes = n_episodes
        self._seed_offset = seed_offset

    def evaluate(self, agent: Agent) -> EvalMetrics:
        shortfalls: list[float] = []
        fill_rates: list[float] = []
        regime_shortfalls: dict[str, list[float]] = defaultdict(list)
        action_counts: dict[int, int] = defaultdict(int)

        for episode in range(self._n_episodes):
            seed = self._seed_offset + episode
            env = self._env_factory(seed)
            obs, _ = env.reset(seed=seed)

            done = False
            info: dict[str, float] = {}
            while not done:
                action = int(agent.act(obs))
                action_counts[action] += 1
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            shortfall = float(info.get("shortfall_so_far", 0.0))
            inventory_remaining = float(info.get("inventory_remaining", 0.0))
            actual_fills = float(info.get("actual_fills", 0.0))
            target_inventory = actual_fills + inventory_remaining
            fill_rate = 0.0 if target_inventory <= 0 else actual_fills / target_inventory
            regime_name = self._REGIME_NAMES.get(int(info.get("regime_id", 1)), "normal")

            shortfalls.append(shortfall)
            fill_rates.append(fill_rate)
            regime_shortfalls[regime_name].append(shortfall)
            env.close()

        total_actions = max(1, sum(action_counts.values()))
        action_dist = {action: count / total_actions for action, count in action_counts.items()}
        for action in range(6):
            action_dist.setdefault(action, 0.0)

        regime_means = {}
        for regime_name in ("calm", "normal", "stressed"):
            values = regime_shortfalls.get(regime_name, [])
            regime_means[regime_name] = float(np.mean(values)) if values else float("nan")

        return EvalMetrics(
            shortfall_mean=float(np.mean(shortfalls)),
            shortfall_std=float(np.std(shortfalls)),
            fill_rate=float(np.mean(fill_rates)),
            regime_shortfall=regime_means,
            action_dist=action_dist,
        )
