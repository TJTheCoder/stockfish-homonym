from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CPP_ROOT = REPO_ROOT / "cpp"
CPP_BUILD_ROOT = REPO_ROOT / "build" / "cpp"
BRIDGE_BINARY = CPP_BUILD_ROOT / "stockfish_platform_bridge"
