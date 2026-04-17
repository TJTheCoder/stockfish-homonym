from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any
import os

import numpy as np

import yaml

from stockfish_homonym.env.platform_execution_env import PlatformEnvConfig, PlatformExecutionEnv
from stockfish_homonym.baselines.twap import PlatformTwapAgent

import stockfish_homonym.learning as learning
from stockfish_homonym.learning import cli_utils
from stockfish_homonym.learning.envs import SequenceEnv, SequenceWrapper
from stockfish_homonym.learning.loading import DiskTrajDataset, get_path_to_trajs


ENV_NAME = "cpp_stock_platform_execution"


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_env(env_config: PlatformEnvConfig, seed: int = 0) -> SequenceEnv:
    return SequenceEnv(
        PlatformExecutionEnv(seed=seed, config=env_config),
        env_name=ENV_NAME,
    )


def bootstrap_twap_buffer(
    cfg: dict[str, Any],
    *,
    run_name: str,
    buffer_dir: str,
    episodes: int,
    seed_offset: int = 0,
) -> int:
    """Populate the protected replay buffer with TWAP demonstrations once."""
    if episodes <= 0:
        return 0

    protected_dir = get_path_to_trajs(buffer_dir, run_name, fifo=False)
    os.makedirs(protected_dir, exist_ok=True)
    existing = [
        name
        for name in os.listdir(protected_dir)
        if name.endswith(".npz") or name.endswith(".traj")
    ]
    if existing:
        return 0

    env_cfg = PlatformEnvConfig(**cfg["env"])
    twap = PlatformTwapAgent(env_cfg.target_inventory, env_cfg.horizon)
    for episode in range(episodes):
        seed = seed_offset + episode
        env = SequenceWrapper(
            make_env(replace(env_cfg, calm_only_episodes=0), seed=seed),
            save_trajs_to=protected_dir,
        )
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            action = twap.act(obs["observation"][0])
            obs, _, terminated, truncated, _ = env.step(
                np.array([action], dtype=np.uint8)
            )
            done = bool(terminated[0] or truncated[0])
        env.save_finished_trajs()
        env.close()
    return episodes


def build_experiment(
    cfg: dict[str, Any],
    *,
    run_name: str,
    buffer_dir: str,
    no_log: bool = True,
    env_mode: str | None = None,
    epochs: int | None = None,
    parallel_actors: int | None = None,
    traj_encoder: str | None = None,
    memory_size: int | None = None,
    memory_layers: int | None = None,
) -> learning.Experiment:
    env_cfg = PlatformEnvConfig(**cfg["env"])
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]

    gin_config: dict[str, Any] = {}
    traj_encoder_type = cli_utils.switch_traj_encoder(
        gin_config,
        arch=traj_encoder or model_cfg["traj_encoder"],
        memory_size=memory_size or model_cfg["memory_size"],
        layers=memory_layers or model_cfg["memory_layers"],
    )
    tstep_encoder_type = cli_utils.switch_tstep_encoder(
        gin_config,
        arch="ff",
        n_layers=model_cfg["tstep"]["n_layers"],
        d_hidden=model_cfg["tstep"]["d_hidden"],
        d_output=model_cfg["tstep"]["d_output"],
        normalize_inputs=False,
    )
    exploration_wrapper_type = cli_utils.switch_exploration(
        gin_config,
        strategy="egreedy",
        eps_start=model_cfg["exploration"]["eps_start"],
        eps_end=model_cfg["exploration"]["eps_end"],
        steps_anneal=model_cfg["exploration"]["steps_anneal"],
        randomize_eps=True,
    )
    agent_type = cli_utils.switch_agent(
        gin_config,
        "agent",
        tau=model_cfg["agent"]["tau"],
        reward_multiplier=model_cfg["agent"]["reward_multiplier"],
        twap_bc_coeff=model_cfg["agent"].get("twap_bc_coeff", 0.0),
    )
    cli_utils.use_config(gin_config)

    dataset = DiskTrajDataset(
        dset_root=buffer_dir,
        dset_name=run_name,
        dset_max_size=train_cfg["dset_max_size"],
    )

    make_train_env = partial(make_env, env_cfg)
    make_eval_env = partial(make_env, replace(env_cfg, calm_only_episodes=0))

    experiment = learning.Experiment(
        run_name=run_name,
        ckpt_base_dir=buffer_dir,
        max_seq_len=train_cfg["max_seq_len"],
        dataset=dataset,
        tstep_encoder_type=tstep_encoder_type,
        traj_encoder_type=traj_encoder_type,
        agent_type=agent_type,
        make_train_env=make_train_env,
        make_val_env=make_eval_env,
        parallel_actors=parallel_actors or train_cfg["parallel_actors"],
        env_mode=env_mode or train_cfg["env_mode"],
        exploration_wrapper_type=exploration_wrapper_type,
        sample_actions_train=True,
        sample_actions_val=False,
        log_to_wandb=not no_log,
        wandb_group_name=run_name,
        traj_save_len=train_cfg["traj_save_len"],
        dloader_workers=train_cfg["dloader_workers"],
        epochs=epochs or train_cfg["epochs"],
        train_timesteps_per_epoch=train_cfg["timesteps_per_epoch"],
        train_batches_per_epoch=train_cfg["batches_per_epoch"],
        val_interval=train_cfg["val_interval"],
        val_timesteps_per_epoch=eval_cfg["val_timesteps"],
        ckpt_interval=train_cfg["ckpt_interval"],
        batch_size=train_cfg["batch_size"],
        mixed_precision=train_cfg["mixed_precision"],
    )
    return experiment
