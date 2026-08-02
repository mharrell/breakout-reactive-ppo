"""
Split-watcher verification — definitive test for argmax reactivity.

Runs the same model on FULL vs ALTERED brick layouts with INDEPENDENT
per-side predictions. Same model weights, same argmax action selection,
two different game states.

A memorized script:  identical paddle positions on both sides.
                     px_corr > 0.99, ALT score ~= FULL score.
A reactive policy:   paddle positions diverge because the ball bounces
                     differently on different brick layouts.

This is the no-timing variant — NoopResetEnv is REMOVED to eliminate
frame-offset timing as a confound. Both sides start at frame 0 identically.
Any divergence comes from different brick layouts affecting ball paths.

Usage:
    python verify_split_watcher.py --model ./models/best_model.zip
    python verify_split_watcher.py --model ./models/best_model.zip --games 60
"""

import sys
import re
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import FireResetEnv, EpisodicLifeEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import cv2
import ale_py
gym.register_envs(ale_py)

BALL_X, BALL_Y, PADDLE_X = 99, 101, 72
NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3

# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class BrickClearWrapper(gym.Wrapper):
    """Clear specified brick RAM addresses on every reset."""
    def __init__(self, env, clear_addrs=None):
        super().__init__(env)
        self._static_addrs = None
        self._random_pct = None
        self._rng = np.random.default_rng()
        if isinstance(clear_addrs, str) and clear_addrs.startswith("random_"):
            self._random_pct = int(clear_addrs.split("_")[1]) / 100.0
        else:
            self._static_addrs = list(clear_addrs or [])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._random_pct is not None:
            n_clear = max(1, int(36 * self._random_pct))
            addrs = list(self._rng.choice(36, size=n_clear, replace=False))
        else:
            addrs = self._static_addrs
        for addr in addrs:
            self.unwrapped.ale.setRAM(addr, 0)
        # Take NOOP step to refresh observation — otherwise both sides
        # see the pre-clear full wall on frame 1.
        obs, _, _, _, _ = self.env.step(0)
        return obs, info


class AutoResetWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            obs, info = self.env.reset()
        return obs, reward, terminated, truncated, info


class GrayscaleResize(gym.ObservationWrapper):
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
# Helpers
# ---------------------------------------------------------------------------

def make_raw_env(brick_addrs=None):
    """Build a raw (non-vec) Breakout env. NO NoopResetEnv — zero timing confound."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = FireResetEnv(env)
    if brick_addrs is not None:
        env = BrickClearWrapper(env, clear_addrs=brick_addrs)
    env = EpisodicLifeEnv(env)
    return env


def get_ram(env):
    return env.unwrapped.ale.getRAM()


def initial_frame_stack(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return [gray] * 4


def update_frame_stack(fs, obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    fs.pop(0)
    fs.append(gray)
    return fs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split-watcher: verify argmax reactivity on Breakout")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model .zip file")
    parser.add_argument("--games", type=int, default=20,
                        help="Games per layout pair (default: 20)")
    args = parser.parse_args()

    MODEL_PATH = args.model
    N_GAMES = args.games
    MAX_FRAMES = 6000

    m = re.search(r"PPO_\d+[a-z]?", MODEL_PATH) or re.search(r"[\w_-]+", MODEL_PATH)
    run_name = m.group(0) if m else "model"

    # Load model with a dummy vec env matching the training pipeline
    def _make_dummy():
        e = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
        e = FireResetEnv(e)
        e = EpisodicLifeEnv(e)
        e = GrayscaleResize(e, width=84, height=84)
        e = AutoResetWrapper(e)
        return e

    dummy_env = DummyVecEnv([_make_dummy])
    dummy_env = VecFrameStack(dummy_env, n_stack=4)
    model = PPO.load(MODEL_PATH, env=dummy_env, device="cuda")
    dummy_env.close()

    print(f"{'='*70}")
    print(f"Split-Watcher Verification — {run_name}")
    print(f"{'='*70}")
    print(f"Model: {MODEL_PATH} @ {model.num_timesteps:,} steps")
    print(f"Inference: deterministic (argmax)")
    print(f"Games per layout pair: {N_GAMES}")
    print()
    print("Principle: INDEPENDENT predictions per side. No NoopResetEnv.")
    print("  Memorized: px_corr > 0.99 AND ALT score ~= FULL score")
    print("  Reactive:  paddle positions DIVERGE on different brick layouts")
    print()

    LAYOUTS = [
        ("RIGHT_HALF", list(range(0, 18))),
        ("LEFT_HALF", list(range(18, 36))),
        ("RANDOM_50", "random_50"),
    ]

    full_scores_all = []
    alt_scores_all = []
    px_corrs_all = []
    divergences_all = []
    perfect_transfers_all = []

    for layout_name, layout_addrs in LAYOUTS:
        for g in range(N_GAMES):
            env_full = make_raw_env(brick_addrs=None)
            env_alt = make_raw_env(brick_addrs=layout_addrs)

            obs_full, _info = env_full.reset()
            obs_alt, _info = env_alt.reset()

            fs_full = initial_frame_stack(obs_full)
            fs_alt = initial_frame_stack(obs_alt)

            full_paddle, alt_paddle = [], []
            full_score, alt_score = 0.0, 0.0
            diverged_frames, compared_frames = 0, 0
            done_full, done_alt = False, False
            step = 0

            while not (done_full and done_alt) and step < MAX_FRAMES:
                step += 1

                # Predict INDEPENDENTLY for each side
                if not done_full:
                    left_obs = np.expand_dims(fs_full, axis=0)
                    left_action, _ = model.predict(left_obs, deterministic=True)
                    left_act = int(left_action[0])
                else:
                    left_act = NOOP

                if not done_alt:
                    right_obs = np.expand_dims(fs_alt, axis=0)
                    right_action, _ = model.predict(right_obs, deterministic=True)
                    right_act = int(right_action[0])
                else:
                    right_act = NOOP

                if not done_full and not done_alt:
                    if left_act != right_act:
                        diverged_frames += 1
                    compared_frames += 1

                # Step FULL side
                if not done_full:
                    try:
                        ram = get_ram(env_full)
                        needs_serve = int(ram[BALL_Y]) > 180
                    except Exception:
                        needs_serve = False
                    act = FIRE if needs_serve else left_act
                    obs, reward, terminated, truncated, info = env_full.step(act)
                    full_score += float(reward)
                    try:
                        px = int(get_ram(env_full)[PADDLE_X])
                    except Exception:
                        px = -1
                    full_paddle.append(px)
                    if terminated or truncated:
                        try:
                            is_game_over = env_full.unwrapped.ale.lives() == 0
                        except Exception:
                            is_game_over = True
                        if is_game_over:
                            done_full = True
                        else:
                            obs, info = env_full.reset()
                            fs_full = [cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)] * 4
                            fs_full = [cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA) for g in fs_full]
                            continue
                    else:
                        update_frame_stack(fs_full, obs)

                # Step ALT side
                if not done_alt:
                    try:
                        ram = get_ram(env_alt)
                        needs_serve = int(ram[BALL_Y]) > 180
                    except Exception:
                        needs_serve = False
                    act = FIRE if needs_serve else right_act
                    obs, reward, terminated, truncated, info = env_alt.step(act)
                    alt_score += float(reward)
                    try:
                        px = int(get_ram(env_alt)[PADDLE_X])
                    except Exception:
                        px = -1
                    alt_paddle.append(px)
                    if terminated or truncated:
                        try:
                            is_game_over = env_alt.unwrapped.ale.lives() == 0
                        except Exception:
                            is_game_over = True
                        if is_game_over:
                            done_alt = True
                        else:
                            obs, info = env_alt.reset()
                            fs_alt = [cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)] * 4
                            fs_alt = [cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA) for g in fs_alt]
                            continue
                    else:
                        update_frame_stack(fs_alt, obs)

            env_full.close()
            env_alt.close()

            # Compute metrics
            div_pct = (diverged_frames / compared_frames * 100) if compared_frames > 0 else 0.0
            min_len = min(len(full_paddle), len(alt_paddle))
            if min_len > 2:
                full_px = np.array(full_paddle[:min_len])
                alt_px = np.array(alt_paddle[:min_len])
                px_corr = np.corrcoef(full_px, alt_px)[0, 1]
            else:
                px_corr = 0.0
            score_ret = (alt_score / full_score * 100) if full_score > 0 else 0.0
            is_perfect_transfer = (px_corr > 0.99 and score_ret > 80)

            full_scores_all.append(full_score)
            alt_scores_all.append(alt_score)
            px_corrs_all.append(px_corr)
            divergences_all.append(div_pct)
            perfect_transfers_all.append(is_perfect_transfer)

            marker = " *** PERFECT TRANSFER ***" if is_perfect_transfer else ""
            print(f"  {layout_name} game {g+1}: {compared_frames}f  |  "
                  f"FULL={full_score:.0f}  ALT={alt_score:.0f} ({score_ret:.0f}%)  |  "
                  f"actions diverged: {diverged_frames}/{compared_frames} ({div_pct:.1f}%)  "
                  f"px_corr={px_corr:.4f}{marker}")

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)

    n_perfect = sum(perfect_transfers_all)
    n_total = len(perfect_transfers_all)
    avg_div = np.mean(divergences_all) if divergences_all else 0
    avg_full = np.mean(full_scores_all) if full_scores_all else 0
    avg_retention = np.mean([a / f * 100 for a, f in zip(alt_scores_all, full_scores_all)
                             if f > 0]) if full_scores_all else 0

    print(f"  Games with perfect transfer (px_corr>0.99, ALT~=FULL): {n_perfect}/{n_total}")
    print(f"  Avg action divergence: {avg_div:.1f}%")
    print(f"  Avg ALT score retention: {avg_retention:.0f}%")
    print(f"  Avg FULL score: {avg_full:.0f}")
    print()

    if avg_full < 5:
        print("  VERDICT: DEAD — policy never learned to play")
    elif n_perfect > 0:
        print(f"  VERDICT: MEMORIZED SCRIPT — {n_perfect}/{n_total} perfect transfers")
        print("  Identical paddle movement on different brick layouts.")
        print("  Physically impossible for a reactive policy.")
    else:
        print(f"  VERDICT: REACTIVE — 0/{n_total} perfect transfers")
        print("  The policy adapts its paddle movement to different brick layouts.")
        print("  No evidence of memorized action sequences.")

    print()
    print("How to read this:")
    print("  Perfect transfer (px_corr>0.99, ALT~=FULL) = DEFINITIVE memorization")
    print("  A reactive policy CANNOT move identically on different brick layouts.")
    print("  This no-timing variant eliminates NoopResetEnv timing as a confound.")
