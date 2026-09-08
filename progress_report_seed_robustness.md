# MAPPO Seed-Robustness Issue — Progress Report

## Background

MAPPO training on the line topology (near/far shelter) shows very different stability depending on random seed. The professor's feedback: if the optimization problem is well-posed, different seeds should converge to the same basin. If they don't, the cause is likely a hyperparameter/problem-conditioning issue — not something to fix by simply training longer or collecting more seeds.

## What was tried

1. **Lowering the learning rate**: compared LR_ACTOR = 3e-4 vs. 1e-4 vs. 3e-5. Spike frequency decreased but did not disappear.
2. **Enlarging the rollout buffer (ROLLOUT_EPISODES 1 → 4)**: confirmed that updating from a single episode's worth of correlated samples alone is enough to cause runaway collapse on its own (seed44 collapsed around ep2280–2300). Averaging over 4 episodes before each update stabilized training substantially — the single most impactful change so far.
3. **Entropy annealing**: built a script that exactly decomposes actor_loss into a `surrogate_term` and an `entropy_term`. This revealed that once the surrogate term shrinks toward ~0 (near equilibrium) later in training, the fixed entropy coefficient comes to dominate the loss/gradient. Changed entropy_coef to anneal from 0.05 → 0.01 anchored to an absolute episode (ep9000) rather than a training-length fraction.
4. **Orthogonal weight initialization**: tried, but in two separate configurations it made the collapse arrive earlier instead of later (rollout=1 + LR=3e-5: crisis moved from ep2400 to ep900; rollout=4 + LR=3e-4: crisis moved from ep7800 to ep1300). Likely explanation: orthogonal init preserves gradient/activation norms more faithfully across layers than the default init, which acts like an effective LR increase — destabilizing rather than stabilizing for this problem. Reverted.
5. **12000-episode run with the best configuration so far** (rollout=4, LR=1e-4, entropy anneal to ep9000, no orthogonal init): fully stable and healthy from ep0–7800 (reward and arrival rate both healthy). However, three periodic crises occurred around ep8500, ep9900, and ep11300 (roughly 1300–1400 episodes apart): reward dropped from ~-5 to -25/-40, arrival time worsened 4–9x, each time self-recovering afterward. Compared to the original unmitigated pattern (785-episode period), frequency roughly halved and onset was delayed, but the instability is not eliminated.

## Mechanistic findings

- Critic loss starts oscillating roughly 1000 episodes before each visible crisis. This looks like an early-warning signal of the critic falling behind a drifting policy ("moving target"), preceding the actual visible collapse in return/arrival time.
- Decomposing actor_loss shows that specifically during the crisis windows (ep8000–11000), the `surrogate_term` (PPO's real policy-improvement signal) flips positive — meaning the batch's average advantage is negative. Since advantage depends on the critic's V(s) estimate, this suggests the critic's baseline is temporarily stale/biased during the drift, systematically distorting the advantage signal rather than the actor's updates being inherently wrong.
- Separately, checking whether the trained policy actually reacts to local congestion when choosing near vs. far shelter showed P(far) barely responds to local density differences (correlation present but tiny effect size). The state itself has meaningful variation (ruled out as an information problem), so the likely cause is that the current reward has the congestion-penalty term explicitly disabled (r_congestion = 0.0), leaving no direct incentive to react to local congestion — only an indirect one via travel-time delay.

## Where we're stuck

- The periodic collapse (moving-target dynamic) is not fully resolved. Next candidate: keep the entropy floor higher (0.01 → 0.02–0.03) rather than annealing it all the way down, to preserve legitimate probabilistic mixing near the congestion-indifference boundary rather than forcing determinism everywhere.
- Also considered giving the critic extra gradient-update epochs per rollout (decoupled from the actor's UPDATE_EPOCHS) so it can catch up faster to a drifting policy, without giving the actor more repeated exposure to the same correlated batch. Not a commonly seen technique, so shelved for now pending advice.
- Considering re-enabling the congestion reward term (currently r_congestion = 0.0), but since this changes the reward design itself, want the professor's input before proceeding.

## Questions for the professor

1. Is an asymmetric design — giving the critic extra update epochs per rollout while leaving the actor's unchanged — a reasonable way to address the moving-target dynamic (critic lag → biased advantage → temporary policy regression)? How is this normally handled?
2. Should the congestion-density reward term be enabled? Right now the policy can only learn to avoid congestion indirectly (through accumulated travel-time delay).
3. More fundamentally, for this environment's congestion-game structure, should the "correct" converged policy allow for state-dependent probabilistic mixing (Wardrop-equilibrium-like), or should it be expected to converge to a deterministic rule?
