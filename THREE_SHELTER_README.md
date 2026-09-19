# Capacity-free three-shelter star experiment

This is the first extension after the frozen `two-shelter-capacity-v1`
checkpoint.  It deliberately changes only the number of independent routes:

- near: 150 m
- middle: 225 m
- far: 300 m
- all roads: 5 m wide
- 3,000 agents, 900 one-second steps
- departure steps uniformly sampled from 0 through 224
- no shelter-capacity limits
- no road blockage

Every agent still makes exactly one irreversible shelter choice.  The choice
and all later rewards are represented as one terminal macro-transition, so the
validated four-episode rollout and PPO update logic remains applicable.

## Network inputs and outputs

- Actor input (4): three normalized road densities and remaining waiting fraction
- Actor output (3): P(near), P(middle), P(far)
- Critic input (7): three normalized densities, three road-occupancy fractions,
  and remaining waiting fraction
- Uniform three-action policy entropy: `ln(3) = 1.098612...`

## Fixed-probability reward landscape

Run this before full training:

```powershell
python evaluate_three_shelter_fixed.py `
  --num-agents 3000 `
  --probability-step 0.1 `
  --evaluation-seeds 100 101 102 103 104 105 106 107 108 109 `
  --output-dir outputs_three_shelter_fixed
```

The sweep evaluates every `(P(near), P(middle), P(far))` grid point whose
probabilities sum to one.  It is an evaluation baseline only and is never
given to the learner.

## Training

```powershell
python three_shelter_training.py `
  --episodes 12000 `
  --num-agents 3000 `
  --entropy-mode annealed `
  --seed 7 `
  --output-dir outputs_three_shelter_annealed
```

Four complete episodes are collected per PPO update, giving 12,000 shelter
choice transitions per normal update and 3,000 updates over 12,000 episodes.

### Rolling checkpoint and resume

Every 25 PPO updates (100 episodes), training atomically replaces one file:

```text
checkpoint_latest.pt
```

No numbered checkpoint history is accumulated. The file is written via a
temporary file and atomic replacement, and it is deleted automatically after
the complete run has saved its final Actor, Critic, diagnostics, and configs.
Resume an interrupted run in place with:

```powershell
python three_shelter_training.py `
  --resume outputs_three_shelter_annealed\run_YYYYMMDD_HHMMSS_annealed_seed7
```

The original total episode target is read from the checkpoint. To deliberately
extend it, also pass a larger `--episodes` value. The checkpoint contains both
networks, both Adam optimizers, the next episode and update counts, diagnostics,
configs, and Python/NumPy/PyTorch/environment random states.

Plot the latest run:

```powershell
python plot_three_shelter_diagnostics.py outputs_three_shelter_annealed
```

## Verification

```powershell
python -m unittest -v test_three_shelter.py
```

The tests cover network shapes, probability normalization, environment action
mapping, four-episode rollout behavior, learning rates, entropy schedule,
one-transition-per-agent collection, finite PPO quantities, and Actor parameter
updates.
