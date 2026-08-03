"""
Minimal PPO training script for Atari Breakout with proximity reward.

Trains a NatureCNN PPO agent on ALE/Breakout-v5 with a dense proximity
reward bonus. Evaluates on clean Breakout (no bonus) to verify transfer.

Usage:
    python train.py                     # train from scratch (25M steps)
    python train.py --resume MODEL.zip  # continue from checkpoint
    python train.py --steps 50000000    # train for 50M steps
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv, NoopResetEnv, FireResetEnv, EpisodicLifeEnv,
)
from proximity_reward_wrapper import ProximityRewardWrapper
import ale_py
gym.register_envs(ale_py)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RUN_NAME = "breakout_proximity"
TARGET_STEPS = 25_000_000
SEED = 42

# Proximity reward parameters (from PPO_124)
PROXIMITY_SCALE = 0.05
PROXIMITY_MAX_DIST = 80.0
PROXIMITY_DESCEND_THRESHOLD = 100

# Standard PPO hyperparameters
ENT_COEF = 0.006
N_ENVS = 32
BATCH_SIZE = 1024
N_STEPS = 128
N_EPOCHS = 4
GAMMA = 0.99


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class GrayscaleResize(gym.ObservationWrapper):
    """Convert RGB to grayscale and resize to 84x84."""
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self._width = width
        self._height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

    def observation(self, obs):
        if obs.ndim == 3 and obs.shape[2] == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(obs, (self._width, self._height),
                             interpolation=cv2.INTER_AREA)
        return resized[:, :, None] if resized.ndim == 2 else resized


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_training_env():
    """Breakout + proximity reward bonus."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = ProximityRewardWrapper(
        env, scale=PROXIMITY_SCALE, max_distance=PROXIMITY_MAX_DIST,
        descend_threshold=PROXIMITY_DESCEND_THRESHOLD,
    )
    env = Monitor(env)
    return env


def make_eval_env():
    """Clean Breakout — no proximity reward. The transfer test."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO on Breakout with proximity reward.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model .zip to resume from")
    parser.add_argument("--steps", type=int, default=TARGET_STEPS,
                        help="Total training steps")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda, cpu, or auto")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"PPO + Proximity Reward — Atari Breakout")
    print(f"{'='*60}")
    print(f"  Proximity bonus: scale={PROXIMITY_SCALE}, "
          f"max_dist={PROXIMITY_MAX_DIST}, "
          f"threshold=ball_y>{PROXIMITY_DESCEND_THRESHOLD}")
    print(f"  Training: ALE/Breakout-v5 + ProximityRewardWrapper")
    print(f"  Eval:     Clean Breakout (no bonus) — transfer test")
    print(f"  PPO:      ent_coef={ENT_COEF}, n_envs={N_ENVS}, "
          f"n_steps={N_STEPS}, n_epochs={N_EPOCHS}")
    print(f"  LR:       2.5e-4 → 1e-5 (linear)")
    print(f"  Clip:     0.2 → 0.05 (linear)")
    print(f"  Target:   {args.steps:,} steps")
    print()

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    # Build vec environments
    env = DummyVecEnv([make_training_env for _ in range(N_ENVS)])
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models",
        log_path="./logs",
        eval_freq=100_000,
        n_eval_episodes=50,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=156_250,
        save_path="./models/checkpoints",
        name_prefix="ppo_proximity",
        save_replay_buffer=False,
        verbose=1,
    )

    # Load or create model
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=env, device=args.device)
        print(f"  Loaded at {model.num_timesteps:,} steps")
    else:
        print("Training from scratch (NatureCNN, random init)")
        model = PPO(
            "CnnPolicy", env,
            learning_rate=get_linear_fn(2.5e-4, 1e-5),
            clip_range=get_linear_fn(0.2, 0.05),
            ent_coef=ENT_COEF,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            seed=SEED,
            device=args.device,
            verbose=1,
        )

    remaining = args.steps - model.num_timesteps
    if remaining <= 0:
        print(f"Target of {args.steps:,} steps already reached.")
    else:
        print(f"Training for {remaining:,} more steps "
              f"({args.steps:,} total target)")
        model.learn(
            total_timesteps=remaining,
            callback=[eval_callback, checkpoint_callback],
            reset_num_timesteps=False,
            tb_log_name=RUN_NAME,
        )

    model.save("./models/final_model")
    print(f"\nTraining complete at {model.num_timesteps:,} steps.")
    print(f"Model saved to ./models/final_model.zip")
    print(f"\nTo verify reactivity, run:")
    print(f"  python verify_split_watcher.py --model ./models/best_model.zip")
    env.close()
    eval_env.close()
