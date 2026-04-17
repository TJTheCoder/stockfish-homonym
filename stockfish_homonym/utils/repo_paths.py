from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
CONFIG_ROOT = PACKAGE_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_ROOT / "default.yaml"
CPP_ROOT = REPO_ROOT / "cpp"
CPP_BUILD_ROOT = REPO_ROOT / "build" / "cpp"
BRIDGE_BINARY = CPP_BUILD_ROOT / "stockfish_platform_bridge"
