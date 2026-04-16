from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from stockfish_homonym.baselines.twap import PlatformTwapAgent
from stockfish_homonym.env.platform_execution_env import PlatformEnvConfig, PlatformExecutionEnv
from stockfish_homonym.eval.evaluator import EvalMetrics, Evaluator
from stockfish_homonym.train.common import build_experiment, load_config

from stockfish_homonym.learning.envs import SequenceEnv, SequenceWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/stockfish_homonym/configs/default.yaml",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--buffer-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=None)
    return parser.parse_args()


def evaluate_policy(
    experiment,
    env_config: PlatformEnvConfig,
    episodes: int,
    seed_offset: int,
) -> EvalMetrics:
    policy = experiment.policy
    policy.eval()
    shortfalls: list[float] = []
    fill_rates: list[float] = []
    regime_shortfalls: dict[str, list[float]] = {"calm": [], "normal": [], "stressed": []}
    action_counts = {action: 0 for action in range(6)}

    regime_map = {0: "calm", 1: "normal", 2: "stressed"}

    for episode in range(episodes):
        seed = seed_offset + episode
        env = SequenceWrapper(
            SequenceEnv(
                PlatformExecutionEnv(seed=seed, config=replace(env_config, calm_only_episodes=0)),
                env_name="cpp_stock_platform_execution",
            ),
            save_trajs_to=None,
        )
        env.reset(seed=seed)
        hidden_state = policy.traj_encoder.init_hidden_state(1, experiment.DEVICE)

        done = False
        info = {}
        while not done:
            obs_np, rl2_np, time_idx_np = env.current_timestep
            obs_t = {
                key: torch.from_numpy(value).to(experiment.DEVICE).unsqueeze(1)
                for key, value in obs_np.items()
            }
            rl2_t = torch.from_numpy(rl2_np).to(experiment.DEVICE).unsqueeze(1)
            time_t = torch.from_numpy(time_idx_np).to(experiment.DEVICE).unsqueeze(1)
            with torch.no_grad():
                actions, hidden_state = policy.get_actions(
                    obs=obs_t,
                    rl2s=rl2_t,
                    time_idxs=time_t,
                    hidden_state=hidden_state,
                    sample=False,
                )
            action = int(actions.squeeze().item())
            action_counts[action] += 1
            _, _, terminated, truncated, info = env.step(np.array([action], dtype=np.uint8))
            done = bool(terminated[0] or truncated[0])

        shortfall = float(info.get("shortfall_so_far", 0.0))
        inventory_remaining = float(info.get("inventory_remaining", 0.0))
        actual_fills = float(info.get("actual_fills", 0.0))
        target_inventory = actual_fills + inventory_remaining
        fill_rate = 0.0 if target_inventory <= 0 else actual_fills / target_inventory
        regime_name = regime_map.get(int(info.get("regime_id", 1)), "normal")
        shortfalls.append(shortfall)
        fill_rates.append(fill_rate)
        regime_shortfalls[regime_name].append(shortfall)
        env.close()

    total_actions = max(1, sum(action_counts.values()))
    action_dist = {action: count / total_actions for action, count in action_counts.items()}
    regime_means = {
        name: float(np.mean(values)) if values else float("nan")
        for name, values in regime_shortfalls.items()
    }
    return EvalMetrics(
        shortfall_mean=float(np.mean(shortfalls)),
        shortfall_std=float(np.std(shortfalls)),
        fill_rate=float(np.mean(fill_rates)),
        regime_shortfall=regime_means,
        action_dist=action_dist,
    )


def format_metrics(label: str, metrics: EvalMetrics) -> str:
    action_summary = "  ".join(
        f"{action}={share:.1%}" for action, share in sorted(metrics.action_dist.items())
    )
    return (
        f"=== {label} ===\n"
        f"  Shortfall mean ± std : {metrics.shortfall_mean:+.4f} ± {metrics.shortfall_std:.4f}\n"
        f"  Fill rate            : {metrics.fill_rate:.2%}\n"
        f"  Regime shortfall     : "
        f"calm={metrics.regime_shortfall['calm']:+.4f}  "
        f"normal={metrics.regime_shortfall['normal']:+.4f}  "
        f"stressed={metrics.regime_shortfall['stressed']:+.4f}\n"
        f"  Actions              : {action_summary}"
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_name = args.run_name or cfg["training"]["run_name"]
    buffer_dir = args.buffer_dir or cfg["training"]["buffer_dir"]
    episodes = args.episodes or cfg["eval"]["episodes"]
    seed_offset = args.seed_offset or cfg["eval"]["seed_offset"]

    experiment = build_experiment(
        cfg,
        run_name=run_name,
        buffer_dir=buffer_dir,
        no_log=True,
        epochs=1,
        parallel_actors=1,
        env_mode="sync",
    )
    experiment.start()
    checkpoint_path = Path(args.checkpoint)
    experiment.load_checkpoint_from_path(str(checkpoint_path), is_accelerate_state=False)

    env_cfg = PlatformEnvConfig(**cfg["env"])
    env_factory = lambda seed: PlatformExecutionEnv(
        seed=seed,
        config=replace(env_cfg, calm_only_episodes=0),
    )
    twap_metrics = Evaluator(
        env_factory=env_factory,
        n_episodes=episodes,
        seed_offset=seed_offset,
    ).evaluate(PlatformTwapAgent(env_cfg.target_inventory, env_cfg.horizon))
    policy_metrics = evaluate_policy(
        experiment=experiment,
        env_config=env_cfg,
        episodes=episodes,
        seed_offset=seed_offset,
    )

    print(format_metrics("Learned policy", policy_metrics))
    print()
    print(format_metrics("TWAP baseline", twap_metrics))
    print()
    delta = twap_metrics.shortfall_mean - policy_metrics.shortfall_mean
    verdict = "better" if delta > 0 else "worse"
    print(f"Policy vs TWAP: {delta:+.4f} ({verdict} than TWAP)")


if __name__ == "__main__":
    main()
