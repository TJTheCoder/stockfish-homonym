from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stockfish_homonym.bridge.client import BridgeClient, BridgeConfig


OBS_DIM = 58


@dataclass(slots=True)
class PlatformEnvConfig:
    target_inventory: int = 250
    horizon: int = 60
    warmup_steps: int = 20
    market_cap: int = 5000
    initial_balance: float = 1_000_000.0
    lambda_risk: float = 0.02
    lambda_urgency: float = 0.5
    calm_only_episodes: int = 0

    def to_bridge_config(self) -> BridgeConfig:
        return BridgeConfig(
            target_inventory=self.target_inventory,
            horizon=self.horizon,
            warmup_steps=self.warmup_steps,
            market_cap=self.market_cap,
            initial_balance=self.initial_balance,
            lambda_risk=self.lambda_risk,
            lambda_urgency=self.lambda_urgency,
        )


class PlatformExecutionEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0, config: PlatformEnvConfig | None = None) -> None:
        super().__init__()
        self._config = config or PlatformEnvConfig()
        self._seed = seed
        self._episode_index = 0
        self._last_info: dict[str, Any] = {}
        self._bridge = BridgeClient(self._config.to_bridge_config())

        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(6)

    @property
    def inventory_remaining(self) -> int:
        return int(self._last_info.get("inventory_remaining", self._config.target_inventory))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        episode_seed = self._seed + self._episode_index if seed is None else seed
        calm_only = self._episode_index < self._config.calm_only_episodes
        payload = self._bridge.reset(seed=episode_seed, calm_only=calm_only)
        self._episode_index += 1
        obs = np.asarray(payload["obs"], dtype=np.float32)
        self._last_info = dict(payload["info"])
        return obs, dict(self._last_info)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        payload = self._bridge.step(int(action))
        obs = np.asarray(payload["obs"], dtype=np.float32)
        reward = float(payload["reward"])
        terminated = bool(payload["terminated"])
        truncated = bool(payload["truncated"])
        info = dict(payload["info"])
        self._last_info = info
        if terminated or truncated:
            info["AMAGO_LOG_METRIC shortfall"] = float(info["shortfall_so_far"])
            info["AMAGO_LOG_METRIC fill_rate"] = float(
                info["actual_fills"] / self._config.target_inventory
            )
            info["AMAGO_LOG_METRIC risk_tolerance"] = float(info["target_risk_tolerance"])
            info["AMAGO_LOG_METRIC regime_id"] = float(info["regime_id"])
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._bridge.close()

    def render(self) -> dict[str, Any]:
        return dict(self._last_info)
