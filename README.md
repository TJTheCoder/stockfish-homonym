# Stockfish Homonym

Stockfish Homonym trains and evaluates a recurrent reinforcement learning agent for stock execution. A C++ simulator handles the market dynamics, Python wraps that simulator as a `gymnasium` environment, and the learned policy is benchmarked against a built-in TWAP baseline.

## Environment Setup

The quickest setup path is the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate stockfish-homonym
```

If you prefer to create the environment manually:

```bash
conda create -n stockfish-homonym python=3.10 cmake
conda activate stockfish-homonym
pip install -e .
```

You still need a working C++ compiler on the machine. The bridge binary is built automatically the first time you run training or evaluation.

## Training The Agent

Run training with the default configuration:

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

Unless overridden, values are loaded from `stockfish_homonym/configs/default.yaml`. By default, training writes checkpoints and replay buffers under `artifacts/stockfish_platform/`. If the protected replay buffer is empty, training first bootstraps it with TWAP demonstration trajectories.

| Option | Default | What it does |
| --- | --- | --- |
| `--config PATH` | packaged `default.yaml` | Load an alternate YAML config file. |
| `--run-name TEXT` | `stockfish_platform` | Name the run and its artifact subdirectory. |
| `--buffer-dir PATH` | `artifacts` | Root directory for checkpoints and replay buffers. |
| `--epochs INT` | `300` | Override the number of training epochs. |
| `--parallel-actors INT` | `4` | Number of rollout actors collecting experience. |
| `--env-mode {sync,async}` | `sync` | Choose synchronous or asynchronous environment stepping. |
| `--traj-encoder TEXT` | `rnn` | Override the trajectory encoder architecture. |
| `--memory-size INT` | `128` | Override the trajectory encoder hidden size. |
| `--memory-layers INT` | `2` | Override the number of trajectory encoder layers. |
| `--log` | off | Enable Weights & Biases logging. |
| `--skip-final-eval` | off | Skip the final validation pass after training. |

## Evaluating A Trained Agent

Evaluate the latest checkpoint against the built-in TWAP baseline:

```bash
stockfish-eval --checkpoint artifacts/stockfish_platform/ckpts/latest/policy.pt
```

You can also point `--checkpoint` at a numbered file inside `ckpts/policy_weights/`. Evaluation runs the learned policy and TWAP on the same seeded episodes and prints mean shortfall, fill rate, regime-level shortfall, action usage, and the policy-vs-TWAP delta.

| Option | Default | What it does |
| --- | --- | --- |
| `--config PATH` | packaged `default.yaml` | Load an alternate YAML config file. |
| `--run-name TEXT` | `stockfish_platform` | Resolve run-scoped defaults such as artifact locations. |
| `--buffer-dir PATH` | `artifacts` | Root directory that contains run artifacts. |
| `--checkpoint PATH` | required | Path to the policy checkpoint to evaluate. |
| `--episodes INT` | `30` | Number of seeded evaluation episodes to run. |
| `--seed-offset INT` | `10000` | Starting seed offset for evaluation episodes. |

## Market Environment Overview

This is an order-execution task, not a stock-picking task. Each episode chooses a target stock in the simulator and asks the agent to buy a fixed inventory within a fixed time horizon while prices, volatility, and market regime evolve underneath it.

With the default config, the environment asks the agent to buy `250` shares over `60` decision steps after a `20`-step market warmup. The simulator models a 10-stock market and moves between calm, normal, and stressed regimes. The reward favors buying below the arrival price while penalizing inventory risk, falling behind schedule, and leaving inventory unfilled by the end of the episode.

### Observation Space

The environment exposes a continuous `Box(low=-5.0, high=5.0, shape=(58,))`.

| Indices | Size | Meaning |
| --- | --- | --- |
| `0` | 1 | Fraction of episode time remaining. |
| `1` | 1 | Fraction of target inventory still unfilled. |
| `2` | 1 | Normalized shortfall so far. |
| `3` | 1 | Normalized cash spent so far. |
| `4` | 1 | Target risk-tolerance bucket. |
| `5` | 1 | Current target price relative to the arrival price. |
| `6` | 1 | Realized volatility of the target stock. |
| `7-8` | 2 | Most recent target-stock returns. |
| `9-11` | 3 | Normalized target-stock ranking under low-, medium-, and high-risk recommenders. |
| `12-14` | 3 | Whether the target stock is the top pick for each recommender. |
| `15` | 1 | Current equity relative to starting balance. |
| `16` | 1 | Normalized unrealized PnL on the target position. |
| `17` | 1 | Urgency score based on remaining inventory vs. remaining time. |
| `18-27` | 10 | Trailing return window for the target stock. |
| `28-37` | 10 | Per-symbol price change vs. each symbol's initial price. |
| `38-47` | 10 | Per-symbol realized volatility features. |
| `48-57` | 10 | One-hot encoding of the current target stock. |

### Action Space

The environment uses `Discrete(6)`. The base child order is `max(1, target_inventory / horizon)`, which is `4` shares under the default config.

| Action | Meaning | Default quantity |
| --- | --- | --- |
| `0` | Wait and place no order. | `0` |
| `1` | Place one base child order. | `4` shares |
| `2` | Place a double-sized child order. | `8` shares |
| `3` | Place a 4x child order. | `16` shares |
| `4` | Place an 8x child order. | `32` shares |
| `5` | Sweep all remaining inventory. | all shares left |
