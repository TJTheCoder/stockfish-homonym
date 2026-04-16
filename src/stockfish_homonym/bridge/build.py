from __future__ import annotations

import subprocess
from stockfish_homonym.utils.repo_paths import BRIDGE_BINARY, CPP_BUILD_ROOT, CPP_ROOT


def _latest_cpp_mtime() -> float:
    latest = 0.0
    for path in CPP_ROOT.rglob("*"):
        if path.is_file():
            latest = max(latest, path.stat().st_mtime)
    return latest


def ensure_bridge_built(force: bool = False) -> None:
    binary_is_fresh = (
        BRIDGE_BINARY.exists()
        and BRIDGE_BINARY.stat().st_mtime >= _latest_cpp_mtime()
    )
    if binary_is_fresh and not force:
        return

    CPP_BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    configure_cmd = ["cmake", "-S", str(CPP_ROOT), "-B", str(CPP_BUILD_ROOT)]
    build_cmd = ["cmake", "--build", str(CPP_BUILD_ROOT), "-j4"]
    subprocess.run(configure_cmd, check=True)
    subprocess.run(build_cmd, check=True)
