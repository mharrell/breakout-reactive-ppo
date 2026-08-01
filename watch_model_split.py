"""
Side-by-side model watcher -- two Breakout games, one window, same agent.

Left side:  standard full-brick Breakout (control)
Right side: different random brick layout each episode

Watch how the agent plays when bricks are where it expects vs. when they
aren't. A memorized script works on the left and implodes on the right.
A reactive policy adapts to both.

Usage:
    python watch_model_split.py --model ./models/PPO_116/best_model.zip
    python watch_model_split.py --model ./models/PPO_115/final_model.zip --games 5 --fps 15 --stoch
    python watch_model_split.py --model ./models/PPO_124/best_model.zip --record
    python watch_model_split.py --model ./models/PPO_124/best_model.zip --record --output ./my_video.mp4
"""
import os
import sys
import time
import re
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import FireResetEnv, NoopResetEnv, EpisodicLifeEnv
import cv2
import ale_py
gym.register_envs(ale_py)

BALL_X, BALL_Y, PADDLE_X = 99, 101, 72
NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3
ACTION_NAMES = ["NOOP", "FIRE", "RIGHT", "LEFT"]

# Brick layout options for the right side -- cycles through these
BRICK_LAYOUTS = [
    ("RIGHT HALF", list(range(0, 18))),       # right bricks removed
    ("LEFT HALF", list(range(18, 36))),        # left bricks removed
    ("TOP HALF", list(range(0, 9)) + list(range(18, 27))),
    ("BOTTOM HALF", list(range(9, 18)) + list(range(27, 36))),
    ("CHECKER", [i for i in range(36) if i % 2 == 0]),
    ("SPARSE 50%", "random_50"),
    ("SPARSE 30%", "random_30"),
]
DISPLAY_W, DISPLAY_H = 480, 320  # per-side display resolution


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
        # Take one NOOP step to refresh the observation — otherwise the
        # returned obs shows the FULL brick wall, not the cleared layout.
        obs, _, _, _, _ = self.env.step(0)
        return obs, info


class GameInstance:
    """One Breakout game: RGB display + grayscale model observation."""

    def __init__(self, brick_addrs=None):
        env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0,
                        render_mode="rgb_array")
        env = NoopResetEnv(env, noop_max=30)
        env = FireResetEnv(env)
        if brick_addrs is not None:
            env = BrickClearWrapper(env, clear_addrs=brick_addrs)
        env = EpisodicLifeEnv(env)
        # NO AutoResetWrapper -- we handle life-loss vs game-over ourselves.
        # AutoResetWrapper resets on every termination, which destroys the
        # information needed to distinguish "lost one life" from "game over."
        self.env = env
        self.frame_stack = []
        self.score = 0.0
        self.frame = 0
        self.done = False
        self.brick_label = "FULL"

    def reset(self):
        obs, info = self.env.reset()
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        self.frame_stack = [gray] * 4
        self.score = 0.0
        self.frame = 0
        self.done = False
        return obs  # RGB for display

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.score += float(reward)
        self.frame += 1

        if terminated or truncated:
            # EpisodicLifeEnv fires terminated=True on EVERY life loss.
            # Only mark truly done when all lives are exhausted (lives==0).
            # For life loss (lives>0), reset the env so the next ball serves.
            try:
                is_game_over = self.env.unwrapped.ale.lives() == 0
            except Exception:
                is_game_over = True  # can't check, trust the env

            if is_game_over:
                self.done = True
            else:
                # Life lost but game continues -- reset for next ball.
                # EpisodicLifeEnv.reset() handles the FIRE-to-serve when
                # was_real_done=False.
                obs, info = self.env.reset()
                # Rebuild frame stack from post-reset obs so the model sees
                # a clean serve, not the death animation.
                gray_reset = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
                gray_reset = cv2.resize(gray_reset, (84, 84),
                                        interpolation=cv2.INTER_AREA)
                self.frame_stack = [gray_reset] * 4
                model_obs = np.stack(self.frame_stack, axis=0)
                return obs, model_obs

        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        self.frame_stack.pop(0)
        self.frame_stack.append(gray)
        model_obs = np.stack(self.frame_stack, axis=0)  # (4, 84, 84)

        return obs, model_obs  # RGB for display, grayscale stack for model


def render_game_text(frame, label, score, ram_data):
    """Overlay game info on the RGB frame."""
    bx, by, px = ram_data
    dx = bx - px
    cv2.putText(frame, f"{label}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 0), 1)
    cv2.putText(frame, f"Score: {int(score)}", (5, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, f"ball=({bx},{by}) pad={px} dx={dx:+d}", (5, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return frame


def make_display_frame(left_rgb, right_rgb, left_label, right_label,
                        left_score, right_score, left_ram, right_ram,
                        frame_num, action_name, is_recording=False):
    """Build the combined side-by-side display frame."""
    # Resize to display resolution
    lf = cv2.resize(left_rgb, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)
    rf = cv2.resize(right_rgb, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)

    lf = render_game_text(lf, left_label, left_score, left_ram)
    rf = render_game_text(rf, right_label, right_score, right_ram)

    # Combine side by side with a divider
    divider = np.full((DISPLAY_H, 2, 3), 128, dtype=np.uint8)
    combined = np.hstack([lf, divider, rf])

    # Top bar with frame info
    top_bar = np.full((24, combined.shape[1], 3), 40, dtype=np.uint8)
    cv2.putText(top_bar, f"Frame: {frame_num}  Action: {action_name}",
                (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    if is_recording:
        # Red recording dot + label
        dot_x = top_bar.shape[1] - 95
        cv2.circle(top_bar, (dot_x, 12), 5, (0, 0, 255), -1)
        cv2.putText(top_bar, "REC", (dot_x + 10, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    return np.vstack([top_bar, combined])


def get_ram(env):
    return env.unwrapped.ale.getRAM()


if __name__ == "__main__":
    MODEL_PATH = "./models/PPO_115/final_model.zip"
    N_GAMES = 10
    FPS = 20
    MODE = "det"
    RECORD = False
    RECORD_OUTPUT = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--model': MODEL_PATH = args[i + 1]; i += 2
        elif args[i] == '--games': N_GAMES = int(args[i + 1]); i += 2
        elif args[i] == '--fps': FPS = int(args[i + 1]); i += 2
        elif args[i] == '--stoch': MODE = "stoch"; i += 1
        elif args[i] == '--det': MODE = "det"; i += 1
        elif args[i] == '--record': RECORD = True; i += 1
        elif args[i] == '--output': RECORD_OUTPUT = args[i + 1]; i += 2
        else: i += 1

    m = re.search(r'PPO_\d+[a-z]?', MODEL_PATH)
    run_name = m.group(0) if m else "model"
    deterministic = MODE == "det"

    print(f"{run_name} side-by-side watch -- {MODE}, {FPS}fps, {N_GAMES} games")
    print(f"  LEFT:  standard Breakout (full bricks)")
    print(f"  RIGHT: random brick layout each game")
    print(f"  Controls: SPACE=pause, ESC=quit, any other key=next game")
    print()

    # Load model — build a dummy env matching the training pipeline exactly
    def _make_dummy():
        e = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
        e = NoopResetEnv(e, noop_max=30)
        e = FireResetEnv(e)
        e = EpisodicLifeEnv(e)
        e = GrayscaleResize(e, width=84, height=84)
        e = AutoResetWrapper(e)
        return e
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    dummy_env = DummyVecEnv([_make_dummy])
    dummy_env = VecFrameStack(dummy_env, n_stack=4)
    model = PPO.load(MODEL_PATH, env=dummy_env, device="cuda")
    dummy_env.close()
    print(f"Loaded {run_name} @ {model.num_timesteps:,} steps")
    print()

    cv2.namedWindow(f"{run_name} -- FULL (L) vs RANDOM BRICKS (R)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"{run_name} -- FULL (L) vs RANDOM BRICKS (R)", DISPLAY_W * 2 + 4, DISPLAY_H + 30)

    # --- Recording setup (deferred until we have the first frame for exact dims) ---
    video_writer = None
    _recorder_needs_init = RECORD
    if RECORD and RECORD_OUTPUT is None:
        os.makedirs("./recordings", exist_ok=True)
        RECORD_OUTPUT = f"./recordings/{run_name}_split_watch.mp4"

    layout_idx = 0
    paused = False
    frame_delay_ms = int(1000 / FPS)

    for game_idx in range(N_GAMES):
        # Pick a random brick layout for the right side
        brick_name, brick_addrs = BRICK_LAYOUTS[layout_idx % len(BRICK_LAYOUTS)]
        layout_idx += 1

        left = GameInstance(brick_addrs=None)   # full bricks
        right = GameInstance(brick_addrs=brick_addrs)

        left.brick_label = "FULL"
        right.brick_label = brick_name

        left_rgb = left.reset()
        right_rgb = right.reset()
        left.done = False
        right.done = False

        print(f"Game {game_idx + 1}: RIGHT = {brick_name}")

        diverged_frames = 0
        total_frames = 0
        left_actions_log = []
        right_actions_log = []

        while not (left.done and right.done):
            if not paused:
                # Predict action INDEPENDENTLY for each side.
                # A reactive policy should pick different actions when the
                # ball is in different positions on the two layouts.
                # A memorized script produces identical actions regardless.

                # Left side prediction
                if not left.done:
                    left_obs = np.expand_dims(left.frame_stack, axis=0)
                    left_action, _ = model.predict(left_obs, deterministic=deterministic)
                    left_act = int(left_action[0])
                else:
                    left_act = NOOP

                # Right side prediction
                if not right.done:
                    right_obs = np.expand_dims(right.frame_stack, axis=0)
                    right_action, _ = model.predict(right_obs, deterministic=deterministic)
                    right_act = int(right_action[0])
                else:
                    right_act = NOOP

                # Step each game independently with its own predicted action.
                # Auto-serve: inject FIRE if ball is in launch zone.
                if not left.done:
                    try:
                        ram = get_ram(left.env)
                        needs_serve = int(ram[BALL_Y]) > 180
                    except Exception:
                        needs_serve = False
                    step_action = FIRE if needs_serve else left_act
                    left_rgb, _ = left.step(step_action)

                if not right.done:
                    try:
                        ram = get_ram(right.env)
                        needs_serve = int(ram[BALL_Y]) > 180
                    except Exception:
                        needs_serve = False
                    step_action = FIRE if needs_serve else right_act
                    right_rgb, _ = right.step(step_action)

                # Track action divergence (exclude serve frames from comparison
                # since FIRE injection is environment-driven, not policy-driven)
                if not left.done and not right.done:
                    left_actions_log.append(left_act)
                    right_actions_log.append(right_act)
                    if left_act != right_act:
                        diverged_frames += 1
                    total_frames += 1

                # Get RAM for overlay
                try:
                    left_ram = (int(get_ram(left.env)[BALL_X]),
                                int(get_ram(left.env)[BALL_Y]),
                                int(get_ram(left.env)[PADDLE_X]))
                except Exception:
                    left_ram = (0, 0, 0)
                try:
                    right_ram = (int(get_ram(right.env)[BALL_X]),
                                 int(get_ram(right.env)[BALL_Y]),
                                 int(get_ram(right.env)[PADDLE_X]))
                except Exception:
                    right_ram = (0, 0, 0)

                # Build and show display — show BOTH actions
                display = make_display_frame(
                    left_rgb, right_rgb,
                    left.brick_label, right.brick_label,
                    left.score, right.score,
                    left_ram, right_ram,
                    left.frame,
                    f"L:{ACTION_NAMES[left_act]} R:{ACTION_NAMES[right_act]}"
                    + (" ***DIVERGE***" if left_act != right_act and not left.done and not right.done else ""),
                    is_recording=RECORD,
                )
                cv2.imshow(f"{run_name} -- FULL (L) vs RANDOM BRICKS (R)", display)
                if _recorder_needs_init:
                    h, w = display.shape[:2]
                    # Try codecs in order: MJPG (.avi) is most portable on Windows
                    for codec_name, ext in [('MJPG', '.avi'), ('mp4v', '.mp4'), ('XVID', '.avi'), ('avc1', '.mp4')]:
                        fourcc = cv2.VideoWriter_fourcc(*codec_name)
                        if RECORD_OUTPUT.endswith('.mp4') and ext != '.mp4':
                            out_path = RECORD_OUTPUT.replace('.mp4', ext)
                        else:
                            out_path = RECORD_OUTPUT
                        video_writer = cv2.VideoWriter(out_path, fourcc, FPS, (w, h))
                        if video_writer.isOpened():
                            RECORD_OUTPUT = out_path
                            print(f"Recording to: {RECORD_OUTPUT}  ({w}x{h} @ {FPS}fps  codec={codec_name})")
                            break
                        video_writer = None
                    if video_writer is None:
                        print(f"WARNING: All codecs failed. Recording disabled.")
                        RECORD = False
                    _recorder_needs_init = False
                if video_writer is not None:
                    video_writer.write(display)

            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if key == 27:  # ESC
                print("\nQuit.")
                if video_writer is not None:
                    video_writer.release()
                    print(f"Saved recording: {RECORD_OUTPUT}")
                cv2.destroyAllWindows()
                sys.exit(0)
            elif key == 32:  # SPACE
                paused = not paused
                if paused:
                    print("  [PAUSED]")
            elif key != 255:  # any other key
                break  # next game

        div_pct = (diverged_frames / total_frames * 100) if total_frames > 0 else 0
        print(f"  FULL={int(left.score):>4}  {brick_name}={int(right.score):>4}"
              f"  |  actions diverged: {diverged_frames}/{total_frames} ({div_pct:.1f}%)"
              f"  {'*** MEMORIZED ***' if div_pct < 2 else '*** REACTIVE? ***' if div_pct > 10 else ''}")

    if video_writer is not None:
        video_writer.release()
        print(f"Saved recording: {RECORD_OUTPUT}")
    cv2.destroyAllWindows()
    print("\nDone.")
