from __future__ import annotations

import argparse

from stockfish_homonym.train.common import build_experiment, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AMAGO agent on the C++ stock platform.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/stockfish_homonym/configs/default.yaml",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--buffer-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--parallel-actors", type=int, default=None)
    parser.add_argument("--env-mode", choices=["sync", "async"], default=None)
    parser.add_argument("--traj-encoder", type=str, default=None)
    parser.add_argument("--memory-size", type=int, default=None)
    parser.add_argument("--memory-layers", type=int, default=None)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_name = args.run_name or cfg["training"]["run_name"]
    buffer_dir = args.buffer_dir or cfg["training"]["buffer_dir"]

    experiment = build_experiment(
        cfg,
        run_name=run_name,
        buffer_dir=buffer_dir,
        no_log=not args.log,
        env_mode=args.env_mode,
        epochs=args.epochs,
        parallel_actors=args.parallel_actors,
        traj_encoder=args.traj_encoder,
        memory_size=args.memory_size,
        memory_layers=args.memory_layers,
    )

    experiment.start()
    experiment.learn()
    if not args.skip_final_eval:
        experiment.evaluate_test(
            experiment.make_val_env,
            timesteps=cfg["eval"]["val_timesteps"],
            episodes=cfg["eval"]["episodes"],
        )


if __name__ == "__main__":
    main()
