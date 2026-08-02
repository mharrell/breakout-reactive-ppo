# Three Lines of Code Fixed 123 Failed PPO Experiments on Atari Breakout

**I spent six months trying to make PPO play Breakout reactively. Every single approach failed — until I tried the simplest possible thing.**

* * *

Collapsed. Memorized. SINGLE_SCRIPT. Not Tracking the Ball. False Positive. I spent six months trying to train a PPO agent to do something I thought would be pretty simple effort for my first "AI" experience: get it to actually play Breakout. But it just wouldn't. It learned a memorized action sequence — a script. The same buttons, in the same order, every game, regardless of where the ball goes. The paddle moves the same way on a full brick wall as it does on a completely different layout. That's not reactive behavior. That's a player piano. Turns out this is the norm and rather expected. I could have taken it as a rookie mistake and tried something new, but I didn't want to. 

I wanted it to work, dangit. Not just like a robot that could input a sequence of movements over and over. But really watch the ball and *play the game.*

If you've ever trained PPO on a deterministic environment and wondered whether your agent was actually *reacting* to what it sees, or just replaying a sequence that worked during training — you're asking the right question. Here's what I found, how I verified it, and the three-line fix that finally made the argmax track the ball.

## The Split-Watcher: A Lie Detector for RL Policies

To do the job right, you need the right tools. Right? My latest tool is called the **split-watcher**.

Here's how it works. You take your trained model and run it on *two* Breakout games side-by-side. The left side has a full brick wall. The right side has a different layout — half the bricks removed, or a checkerboard pattern, or 50% random. Both sides are driven by the **same model** making **independent predictions**. Same weights, same argmax action selection, two different game states.

The key insight: **different brick layouts cause different ball bounces.** On the right side, the ball hits different bricks at different angles. It arrives at the paddle from different positions. A reactive policy that actually tracks the ball *must* move differently on the two sides. It has no choice — the ball is somewhere else.

A memorized script, on the other hand, plays the same action sequence regardless. The paddle ends up in identical positions on both layouts. Compute the Pearson correlation of paddle positions: **px_corr > 0.99 means definitive memorization.** Physically impossible for a reactive policy.

This test measures what your agent actually *does* — the argmax action — not what its probability distribution *thinks about doing*. That distinction turns out to matter a lot.

[Watch the split-watcher in action ->](https://www.youtube.com/watch?v=6ixVwQm7u5Y)

## 123 Ways to Fail

Over the course of this project, I tried systematically eliminating memorization from every angle. Each experiment varied one thing while holding everything else constant. Every single one was memorized. I kept thinking, when we teach students in school, we don't "teach for the test" we teach them good principles and problem solving skills, and then test them later. This was the approach I kept trying to take, to no avail. The scripts either just failed to find any success at all or would become so robust they could function despite everything I threw at it.

Here's the graveyard:

<table>
<tr><th>Approach</th><th>What I Tried</th><th>Why It Failed</th></tr>
<tr><td>Sticky actions (p=0.25)</td><td>Random action noise during training</td><td>Breakout is forgiving — scripts survive 25% noise. Dead policy + sticky = 8–14 "unique" scores. The noise masks memorization; it doesn't prevent it. Sticky actions were proposed by Machado et al. (2018) as the standard ALE evaluation protocol, but Zhang et al. (2018) had already shown they don't prevent memorization in deep RL — a finding this project independently confirmed: every sticky-trained model collapsed to a deterministic script without sticky actions.</td></tr>
<tr><td>Cursor wrappers (13 variants)</td><td>Adversarial "cursor" that attacks the paddle when it's far from the ball</td><td>PPO learned to hedge: maintain a reactive-looking distribution (where's the ball?) while the argmax converged to a fixed script. The distribution tracks the ball; the mode ignores it.</td></tr>
<tr><td>Entropy bonuses (0.006–0.10)</td><td>Reward the policy for diverse action distributions</td><td>Entropy came from random action noise, not from tracking the ball.</td></tr>
<tr><td>Frame skip</td><td>Unpredictable observation timing</td><td>The CNN conditioned on the skip pattern. PPO found a skip-conditioned script.</td></tr>
<tr><td>Dynamics randomization</td><td>Vary ball physics via setRAM() per episode</td><td>CNN conditioned on the first few frames. Layout-conditioned, episode-conditioned — not reactive.</td></tr>
<tr><td>Random bricks each episode</td><td>Different brick layout every reset</td><td>Same — conditioned on initial observation, not tracking.</td></tr>
<tr><td>Trajectory entropy</td><td>Reward taking <i>different</i> actions across parallel environments at the same frame</td><td>Scripts with timing offsets produce different actions at the same frame. Superficial diversity.</td></tr>
<tr><td>Moving bumpers (15 shapes)</td><td>Indestructible brick shapes that reposition every 60–150 frames</td><td>Ball-bounce timing variance produced false score diversity.</td></tr>
<tr><td>Extreme bumper (2 independent)</td><td>Two bumpers with 3+ brick shapes each</td><td>Same problem.</td></tr>
<tr><td>One-life training</td><td>No EpisodicLifeEnv — harder penalty for mistakes</td><td>Scripts still viable. You can script Breakout with one life.</td></tr>
<tr><td>Brick pre-clearing</td><td>Start each episode with 15–25 random bricks already cleared</td><td>CNN conditioned on remaining bricks. Another conditioned script.</td></tr>
<tr><td>Ball-binned trajectory entropy</td><td>Reward action diversity conditioned on ball position (LEFT/CENTER/RIGHT bins)</td><td>Timing offsets still produced false diversity signals.</td></tr>
<tr><td>Random ball bounce perturbation</td><td>Nudge ball X by N(0, 3) on every paddle bounce</td><td>Breakout is STILL forgiving enough for scripts to survive unpredictable ball trajectories.</td></tr>
<tr><td>Auxiliary ball-position supervision</td><td>Add regression head: predict ball (x, y) from conv features</td><td>CNN learned to locate the ball with 1.9px precision. Policy still memorized. Features baked in, argmax ignored them.</td></tr>
<tr><td>BeamRider (different game!)</td><td>Hard failure — one mistake kills you, as well as adversarial enemies that attack you and must be fought off</td><td>Didn't help. BeamRider models were SINGLE_SCRIPT too (std=0.0). Same distribution-vs-argmax confound, different game.</td></tr>
</table>

Look at that list again. Every single approach tried to **penalize scripts** — make the environment harder to memorize, make the objective function punish repetitive behavior, make the architecture attend to the right things. PPO's optimizer found a way around every single one. It always converged on a script because **in a deterministic environment, a script maximizes expected return.**

The optimizer isn't broken. We're asking it to optimize the wrong thing.

## The Fix, Three Lines at a Time

After 123 experiments and months of brainstorming, false positives, faulty diagnostics, and even building my own custom Breakout engine at one point, I tried something obvious:

**What if I just... rewarded the paddle for being near the ball?**

Not hitting the ball. Not scoring points. Not survival. Not penalizing scripts. Not making the environment harder. Not trying to trick the optimizer. Just directly rewarding the behavior I actually wanted.

<pre><code>def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)

    ball_y = self._get_ram(101)          # ball Y position
    if ball_y > 100:                      # ball descending toward paddle
        paddle_x = self._get_ram(72)      # paddle X position
        ball_x = self._get_ram(99)        # ball X position
        distance = abs(paddle_x - ball_x)
        bonus = 0.05 * max(0.0, 1.0 - distance / 80.0)
        reward += bonus                    # &lt;- that's it

    return obs, reward, terminated, truncated, info</code></pre>

Three lines. A reward of up to 0.05 per frame — **twenty frames of perfect tracking equals one yellow brick.** On an average game with ~2,000 descent frames, perfect tracking earns about 50 bonus points. That's roughly 5–10% of the total game score.

The bonus is only applied during training. Evaluation uses clean Breakout with no bonus — the behavior has to transfer.

## Why This Works When Everything Else Didn't

The key difference is in what gets optimized.

Every previous approach tried to change the **viability of scripts.** Make the environment stochastic so scripts fail. Add an entropy penalty so repetitive sequences are punished. Or somehow encourage the natural experience of human play. The problem: PPO always found a script that survived the changes. A timing-robust script. A layout-conditioned script. A noise-tolerant script. The *optimum* was still a script — only the shape changed.

Proximity reward changes **what the optimum is.**

A center-hold script gets *some* proximity reward incidentally — the ball passes near center on some descents. But a reactive tracking policy gets the **maximum bonus on every single descent frame.** Over a full game, the tracker earns 3–5× more proximity bonus than the center-hold script. The optimization pressure is unambiguous: **track the ball, get more reward.** There's no clever script that can fake being close to the ball.

The bonus is dense (every frame) while the game's natural reward — brick breaks — is sparse (every few seconds). Dense rewards provide better gradients. The model doesn't need to discover the multi-step causal chain "track ball -> hit ball -> ball breaks brick -> score." It's told directly: being near the ball is good.

And because the bonus is so small (0.05 vs 1.0–7.0 per brick), it doesn't overwhelm the game reward. The model still cares about breaking bricks. It just learns that the way to break bricks is to be under the ball when it comes down.

## The Evidence

### The Split-Watcher: 0/240 Perfect Transfers

I ran 240 split-watcher games across two checkpoints (19.2M and 25M steps) and three layout types. **Zero perfect transfers.** No game showed identical paddle movement on different brick layouts.

When I eliminated NoopResetEnv timing offsets (both sides start at frame 0), the model **cleared every brick on every layout, every game.** 120 out of 120. It doesn't matter which bricks you remove — the policy adapts and clears what's there.

### The Intervention Gradient: Clean Dose-Response

I built a test that teleports the ball horizontally by varying amounts (0–60 pixels) mid-game and checks whether the model reverses its paddle direction to follow. A memorized script ignores the teleport. A reactive tracker changes direction.

The dead baseline (center-hold script) scored 0.0% reversal at all magnitudes. The final model peaked at **60.0% reversal** at 15px displacement, declining smoothly at larger displacements where the ball is in physically impossible positions. The dose-response curve (AUC = 0.421) is classified as STRONG — and it has a clean shape that memorized models never show.

### The Numbers (clean eval, no proximity reward)

<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Best stochastic score</td><td><b>216</b> (highest ever on clean Breakout in this project with only 25 million training steps)</td></tr>
<tr><td>det=True MULTIPLE_SCRIPTS</td><td>10 of last 12 checkpoints (first time without sticky masking)</td></tr>
<tr><td>FULL-wall deterministic score</td><td>379–383</td></tr>
<tr><td>No-timing ALT retention</td><td><b>100%</b> (clears every layout)</td></tr>
<tr><td>Intervention AUC</td><td><b>0.421</b> (STRONG)</td></tr>
</table>

## What This Means for Your PPO Projects

If you're training PPO on any deterministic environment, there's a good chance your argmax is a script. The policy distribution might look reactive — it might shift its probabilities in response to game state. Your eval scores might look great. But the action actually *taken* could be a fixed sequence that ignores what's happening on screen.

The split-watcher gives you a way to check. Run your model on two different initial states. Give it independent predictions. If it moves identically on both, it's not reacting — it's replaying.

And if you find memorization, the fix might not be more environment engineering. It might be simpler than you think. **Reward what you want. Don't penalize what you don't want.** PPO will optimize whatever you put in the reward function. Give it the right target.

## Try It Yourself

<pre><code>git clone https://github.com/mharrell/breakout-reactive-ppo.git
cd breakout-reactive-ppo
pip install -r requirements.txt

# Train (25M steps, ~6–7 hours on RTX 3060 Ti)
python train.py

# Verify reactivity
python verify_split_watcher.py --model ./models/best_model.zip --games 20

# Watch it live
python watch_model_split.py --model ./models/best_model.zip --record</code></pre>

Everything you need is in the repo: the wrapper, the training script, the split-watcher, and the data. No dependencies on the parent project. Clean, minimal, reproducible.

The full history — all 123 failures, the diagnostic methodology, the blind spots I discovered along the way — lives at [BreakoutBot](https://github.com/mharrell/BreakoutBot). It's a messy, honest record of what it looks like to systematically eliminate hypotheses until only one is left standing.

* * *

*Mike Harrell is an independent ML researcher. He does not have a master's degree or a PhD, but if he did, you can bet he'd be the kind of guy that would shout "Just what the doctor ordered!" every time he opened a package in the mail. You can find the code at [github.com/mharrell/breakout-reactive-ppo](https://github.com/mharrell/breakout-reactive-ppo) and the split-watcher video at [youtube.com/watch?v=6ixVwQm7u5Y](https://www.youtube.com/watch?v=6ixVwQm7u5Y).*

* * *

*P.S. — I'm trying to get this posted on arXiv as a standalone paper so it's citable and findable. If you're an active arXiv author in cs.AI or cs.LG and found this work useful, you can endorse me here: https://arxiv.org/auth/endorse?x=MUM8BP — it takes one click. Or email me at mikey.harrell@gmail.com. Thanks for reading.*
