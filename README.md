# Stockfish Homonym

This codebase trains and evaluates a recurrent reinforcement learning agent for stock execution. A C++ simulator handles the market side, Python wraps it as a `gymnasium` environment, and the training code learns a policy and compares it against a TWAP baseline.

## Setup And Run

Create a Python 3.10 environment, activate it, and install the package:

```bash
conda create -n stockfish-homonym python=3.10
conda activate stockfish-homonym
pip install -e .
```

That install pulls in the Python training stack and CMake. You still need a working C++ compiler on the machine. The bridge is built automatically the first time you run training. Start a run with:

```bash
stockfish-train
```

Example with a few overrides:

```bash
stockfish-train \
  --run-name my_run \
  --buffer-dir artifacts \
  --epochs 50 \
  --parallel-actors 2 \
  --env-mode sync
```

## Evaluate

Evaluate a saved checkpoint against the built-in TWAP baseline with:

```bash
stockfish-eval --checkpoint artifacts/amago_cpp_platform/ckpts/latest/policy.pt
```

You can also point that command at one of the numbered files in `ckpts/policy_weights/`.
