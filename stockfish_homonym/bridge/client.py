from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any

from stockfish_homonym.bridge.build import ensure_bridge_built
from stockfish_homonym.utils.repo_paths import BRIDGE_BINARY


@dataclass(slots=True)
class BridgeConfig:
    target_inventory: int = 250
    horizon: int = 60
    warmup_steps: int = 20
    market_cap: int = 5000
    initial_balance: float = 1_000_000.0
    lambda_risk: float = 0.02
    lambda_urgency: float = 0.5


class BridgeClient:
    def __init__(self, config: BridgeConfig) -> None:
        self._config = config
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._process is not None:
            return

        ensure_bridge_built()
        args = [
            str(BRIDGE_BINARY),
            "--target-inventory",
            str(self._config.target_inventory),
            "--horizon",
            str(self._config.horizon),
            "--warmup-steps",
            str(self._config.warmup_steps),
            "--market-cap",
            str(self._config.market_cap),
            "--initial-balance",
            str(self._config.initial_balance),
            "--lambda-risk",
            str(self._config.lambda_risk),
            "--lambda-urgency",
            str(self._config.lambda_urgency),
        ]
        self._process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def close(self) -> None:
        if self._process is None:
            return
        try:
            self._send_command("CLOSE")
        except RuntimeError:
            pass
        finally:
            if self._process.poll() is None:
                self._process.kill()
            self._process.wait(timeout=2)
            if self._process.stdin is not None:
                self._process.stdin.close()
            if self._process.stdout is not None:
                self._process.stdout.close()
            if self._process.stderr is not None:
                self._process.stderr.close()
            self._process = None

    def reset(self, seed: int, calm_only: bool) -> dict[str, Any]:
        return self._send_command(f"RESET {seed} {1 if calm_only else 0}")

    def step(self, action: int) -> dict[str, Any]:
        return self._send_command(f"STEP {action}")

    def _send_command(self, command: str) -> dict[str, Any]:
        self.start()
        assert self._process is not None
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("Bridge process pipes are unavailable.")

        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            stderr = ""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read()
            raise RuntimeError(
                f"Bridge process exited unexpectedly while running `{command}`. {stderr}".strip()
            )

        payload = json.loads(line)
        if "error" in payload:
            raise RuntimeError(payload["error"])
        return payload

    def __enter__(self) -> "BridgeClient":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
