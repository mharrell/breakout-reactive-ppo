"""
ProximityRewardWrapper — directly reward keeping the paddle near the ball.

In deterministic Atari Breakout, PPO converges to a memorized action sequence
rather than a reactive ball-tracking policy. This wrapper adds a per-step bonus
proportional to horizontal paddle-ball proximity during descent:

    distance = |paddle_x - ball_x|
    bonus = scale × max(0, 1 - distance / max_distance)

When ball_y > descend_threshold (ball descending toward paddle): bonus applied.
When ball_y <= descend_threshold (ball in brick zone): no bonus.

The bonus is tiny (scale=0.05, ~50 pts/game) compared to brick breaks (1-7 pts),
but it's dense — it fires every frame. This provides a continuous gradient
toward ball-tracking that doesn't require multi-step credit assignment.

Transfer test: evaluate on CLEAN Breakout (no wrapper). If ball-tracking
persists, the policy learned a general behavior, not just bonus-maximization.

RAM addresses (ALE 0.11.2, Breakout ROM):
  RAM 72: paddle_x (0-160ish)
  RAM 99: ball_x   (0-199, playfield ~0-160)
  RAM 101: ball_y  (0-210ish)
"""

import numpy as np
import gymnasium as gym


class ProximityRewardWrapper(gym.Wrapper):
    """Reward horizontal proximity between paddle and ball during descent.

    Parameters
    ----------
    scale : float
        Maximum per-step bonus when paddle is exactly at ball_x (default 0.05).
    max_distance : float
        Horizontal distance where bonus reaches zero (default 80).
        |paddle_x - ball_x| >= 80 → bonus = 0.
    descend_threshold : int
        Only apply bonus when ball_y > this value (default 100).
    """

    PADDLE_X_ADDR = 72
    BALL_X_ADDR = 99
    BALL_Y_ADDR = 101

    def __init__(self, env, scale=0.05, max_distance=80.0, descend_threshold=100):
        super().__init__(env)
        self.scale = float(scale)
        self.max_distance = float(max_distance)
        self.descend_threshold = int(descend_threshold)
        self._total_bonus = 0.0

    def _get_ram(self, addr):
        return int(self.env.unwrapped.ale.getRAM()[addr])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        ball_y = self._get_ram(self.BALL_Y_ADDR)
        if ball_y > self.descend_threshold:
            paddle_x = self._get_ram(self.PADDLE_X_ADDR)
            ball_x = self._get_ram(self.BALL_X_ADDR)
            distance = abs(paddle_x - ball_x)
            bonus = self.scale * max(0.0, 1.0 - distance / self.max_distance)
            reward += bonus
            self._total_bonus += bonus

        if terminated or truncated:
            if isinstance(info, dict):
                info["proximity_bonus"] = self._total_bonus

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._total_bonus = 0.0
        return self.env.reset(**kwargs)
