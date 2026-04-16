from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SISTER_ROOT = REPO_ROOT.parent
AMAGO_ROOT = SISTER_ROOT / "amago"
CPP_ROOT = REPO_ROOT / "cpp"
CPP_BUILD_ROOT = REPO_ROOT / "build" / "cpp"
BRIDGE_BINARY = CPP_BUILD_ROOT / "stockfish_platform_bridge"


def ensure_amago_on_path() -> None:
    amago_path = str(AMAGO_ROOT)
    if amago_path not in sys.path:
        sys.path.insert(0, amago_path)
