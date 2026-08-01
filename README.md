# Three Lines of Code Fixed 123 Failed PPO Experiments

**PPO on Atari Breakout converges to a memorized action sequence, not a reactive ball-tracking policy. Here's why, and here's the fix.**

---

## The Problem

In deterministic environments like Atari Breakout, PPO's objective function is:

$$ \text{argmax}_\pi \ \mathbb{E}\left[\sum \text{rewards}\right] $$

A memorized action sequence (script) is a *valid* policy — and in a deterministic environment, it often maximizes expected return better than a reactive policy. PPO's optimizer isn't broken. It's doing exactly what we asked.

The result: after training, the argmax produces the same action sequence every game regardless of where the ball is. The paddle moves identically on different brick layouts — physically impossible for a reactive policy.

**123 experiments** tried to fix this by making scripts non-viable:
sticky actions, cursor wrappers, entropy bonuses, frame skip, dynamics randomization,
adversarial bumpers, brick randomization, ball perturbations, trajectory entropy,
one-life training, moving bumpers, frozen features, auxiliary supervision...

Every single one failed. The argmax was always a script.

## The Fix

Stop penalizing scripts. Start **rewarding tracking**:

```python
# proximity_reward_wrapper.py
def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)

    ball_y = self._get_ram(self.BALL_Y_ADDR)       # RAM 101
    if ball_y > 100:                                 # ball descending
        paddle_x = self._get_ram(self.PADDLE_X_ADDR) # RAM 72
        ball_x = self._get_ram(self.BALL_X_ADDR)     # RAM 99
        distance = abs(paddle_x - ball_x)
        bonus = 0.05 * max(0.0, 1.0 - distance / 80.0)
        reward += bonus

    return obs, reward, terminated, truncated, info
```

That's it. Three lines. A tiny bonus (0.05 per frame vs 1.0–7.0 per brick) for keeping the paddle horizontally close to the ball during descent. Dense, consistent, unambiguous.

The bonus is only applied during **training**. Evaluation uses clean Breakout with no bonus — the ball-tracking behavior transfers.

## Why It Works

Every previous approach tried to **penalize scripts** by making the environment harder to memorize. PPO's optimizer always found a way around the penalty — a different script, a timing-robust script, a layout-conditioned script. The optimum was always a script; only the shape changed.

Proximity reward changes the **optimum itself**. Now the reward-maximizing behavior is ball-tracking, not script-execution. A center-hold script gets some incidental proximity reward when the ball happens to pass near center. A reactive tracker gets the maximum bonus on every descent frame. The gradient is unambiguous.

The bonus is dense (every frame) while game rewards are sparse (bricks break every few seconds). Dense rewards provide better gradients for credit assignment. The model doesn't need to discover that tracking → hitting → scoring — it's told that tracking IS rewarding.

## The Split-Watcher: How We Know It's Real

Every diagnostic in the standard RL toolbox measures the **policy distribution** — the probability of each action. But evaluation uses the **argmax** — the single highest-probability action. A model can maintain a reactive-looking distribution (90% LEFT when ball is left, 90% RIGHT when ball is right) while the argmax is always RIGHT regardless.

The split-watcher measures the argmax directly:

```
         FULL WALL                    CLEARED (RIGHT HALF)
    ┌──────────────────┐          ┌──────────────────┐
    │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │          │                  │
    │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │          │                  │
    │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │          │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
    │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │          │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
    │        ●          │          │        ●          │
    │       ───         │          │       ───         │
    └──────────────────┘          └──────────────────┘
      Same model, same weights, DIFFERENT predictions per side
```

**A memorized script:** identical paddle positions on both sides (px_corr > 0.99).
**A reactive policy:** paddle positions diverge because different bricks → different ball bounces → different tracking responses.

We run 60 games across three layout types (right half cleared, left half cleared, 50% random). A single game with px_corr > 0.99 and ALT score ≈ FULL score means **definitive memorization** — physically impossible for a reactive policy.

## Results

### Split-Watcher (NoopResetEnv removed — zero timing confound)

| Checkpoint | Games | Perfect Transfers | ALT Score Retention | Action Divergence |
|-----------|-------|-------------------|---------------------|-------------------|
| best (19.2M) | 60 | **0** | **100%** | 62.4% |
| final (25M) | 60 | **0** | **100%** | 62.9% |

**0/120 perfect transfers. The model clears every brick on every layout, every game.**

### Split-Watcher (with NoopResetEnv — realistic 0–30 frame offset at reset)

| Checkpoint | Games | Perfect Transfers | ALT Score Retention | Action Divergence |
|-----------|-------|-------------------|---------------------|-------------------|
| best (19.2M) | 60 | **0** | 46% | 71.4% |
| final (25M) | 60 | **0** | 59% | 70.6% |

**0/120 perfect transfers. 240 games total. Zero.**

Every prior model (PPO_111–118, BeamRider) had at least one perfect transfer. This is the first model to score zero across all tests.

### Memorization Check (det=True, clean eval — no proximity reward)

| Checkpoint | Unique Scores | Best Score | Verdict |
|------------|:---:|:---:|---------|
| 1M | 1 | 16 | SINGLE_SCRIPT |
| 5M | 1 | 37 | SINGLE_SCRIPT |
| 10M | 1 | 78 | SINGLE_SCRIPT |
| 14M | **4** | **87** | **MULTIPLE_SCRIPTS** |
| 19M | 2 | 107 | SINGLE_SCRIPT |
| 25M | **4** | 93 | **MULTIPLE_SCRIPTS** |

First model in project history to sustain MULTIPLE_SCRIPTS on det=True without sticky masking.

### Intervention Gradient (dose-response: does the policy reverse direction when the ball is teleported?)

| Magnitude | Dead Baseline | final (25M) |
|-----------|:---:|:---:|
| ±0 px | 0.0% | 37.5% |
| ±8 px | 0.0% | 41.2% |
| ±15 px | 0.0% | **60.0%** |
| ±30 px | 0.0% | 50.0% |
| ±45 px | 0.0% | 31.2% |
| ±60 px | 0.0% | 25.0% |
| **AUC** | 0.000 | **0.421** |

Clean dose-response curve: peaks at moderate displacement (the ball moved), drops at extreme displacement (ball in physically impossible position). Dead baseline: 0.0% at all magnitudes. AUC 0.421 classified as STRONG.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train (25M steps, ~8–12 hours on RTX 3060 Ti)
python train.py

# Verify reactivity
python verify_split_watcher.py --model ./models/best_model.zip --games 20

# Watch it live (side-by-side, two layouts, same agent)
python watch_model_split.py --model ./models/best_model.zip --record
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Stable-Baselines3 2.3+
- ALE 0.11+
- GPU with 8GB+ VRAM (RTX 3060 Ti or better)
- ~8–12 hours for 25M steps on a single GPU

## What We Learned

1. **Reward what you want, don't penalize what you don't want.** 123 experiments tried to make scripts non-viable. The fix was directly rewarding the desired behavior.

2. **Dense rewards beat sparse rewards for shaping behavior.** Brick breaks are sparse (every few seconds) and require multi-step credit assignment. The proximity bonus fires every frame.

3. **Scale doesn't need to be large if the signal is consistent.** 0.05 × 2,000 frames ≈ 50 bonus per game (~7 yellow bricks). That's enough.

4. **The argmax follows the reward, not the distribution.** The cursor wrapper shaped the distribution but the argmax converged to the mode. Proximity reward makes tracking the highest-value action.

5. **PPO's objective function was the root cause.** `argmax_π E[Σ rewards]` in a deterministic environment converges to a script because scripts maximize expected return. Change the reward, not the environment.

## The Full Story

This finding emerged from [BreakoutBot](https://github.com/mharrell/BreakoutBot), a solo research project that ran 124 controlled PPO experiments on Atari Breakout over several months. Each experiment varied one variable while holding everything else constant. The full history — all 123 failures, the diagnostic blind spots discovered along the way, the split-watcher methodology, and the eventual breakthrough — is documented in that repository.

This repo is the clean, minimal reproduction. No baggage, no dead experiments. Just the wrapper, the training script, and the verification tool.

## Citation

If you use this work, please cite:

```bibtex
@misc{harrell2025proximity,
  title={Dense Reward Shaping Produces Reactive PPO Policies in Atari Breakout},
  author={Harrell, Mike},
  year={2025},
  howpublished={\url{https://github.com/mharrell/breakout-reactive-ppo}},
}
```

## License

MIT
