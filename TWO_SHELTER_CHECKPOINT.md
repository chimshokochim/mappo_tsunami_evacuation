# Two-shelter capacity-aware MAPPO checkpoint

This document records the reproducible state of the two-shelter experiment
before development moves to a larger network.

## Source checkpoint

- Repository: `https://github.com/chimshokochim/mappo_tsunami_evacuation`
- Stable implementation commit: `5d625be`
- Checkpoint tag: `two-shelter-capacity-v1`
- Recorded: 2026-09-18

## Model and environment

- 3,000 agents, 900 one-second steps
- departure steps sampled uniformly from 0 through 224
- near road: 150 m x 5 m
- far road: 300 m x 5 m
- per-agent free-flow speed: 1.0--1.5 m/s
- speed is unchanged below density 0.1 agents/m2, decreases linearly, and is
  0.3 times base speed at or above density 1.0 agents/m2
- congestion penalty coefficient: 0.50
- time penalty: -0.01 per active step
- capacity failure penalty: -500 per failed agent
- no road blockage in the capacity experiment
- reject-mode shelter capacity handling
- near capacity sampled per episode from
  2400, 2300, 2200, 2100, 2000, 1900, and 1800
- far capacity fixed at 2400

Actor inputs (7):

1. normalized near density
2. normalized far density
3. remaining waiting fraction
4. remaining near capacity / 3000
5. remaining far capacity / 3000
6. remaining near capacity / remaining waiting agents
7. remaining far capacity / remaining waiting agents

Critic inputs (9) use the same seven features plus the fractions of agents
currently travelling on the near and far roads.

## PPO configuration

- Actor: 7 -> 64 -> 64 -> 2, Tanh/Tanh/Softmax
- Critic: 9 -> 64 -> 64 -> 1, Tanh/Tanh/linear
- Actor learning rate: 1e-4
- Critic learning rate: 1e-3
- gamma: 0.99
- GAE lambda stored in config: 0.95
- each one-time shelter choice is represented as a terminal macro-transition
- advantage: system-wide mean discounted reward-to-go minus critic value
- rollout: four complete episodes (normally 12,000 transitions)
- PPO epochs: 4
- minibatch size: 512
- clipping epsilon: 0.2
- maximum gradient norm: 0.5
- entropy coefficient: 0.05 to 0.01 linearly through episode 9,000, then 0.01

The saved `environment_config.json` and `ppo_config.json` inside every run
directory are the authoritative configuration records.

## Trained runs

- seed 7:
  `outputs_capacity_reject_annealed/run_20260915_144303_641676_annealed_seed7`
- seed 17:
  `outputs_capacity_reject_seed_sweep/run_20260917_150111_948995_annealed_seed17`
- seed 27:
  `outputs_capacity_reject_seed_sweep/run_20260917_173032_501494_annealed_seed27`
- seed 37:
  `outputs_capacity_reject_seed_sweep/run_20260917_210659_954712_annealed_seed37`

Each run directory contains Actor and Critic weights, diagnostics, environment
and PPO configuration JSON files, and run metadata.

## Reproduction commands

Train the first seed:

```powershell
python training.py `
  --episodes 12000 `
  --seed 7 `
  --entropy-mode annealed `
  --output-dir outputs_capacity_reject_annealed `
  --near-capacities 2400 2300 2200 2100 2000 1900 1800 `
  --far-capacity 2400 `
  --failure-penalty -500 `
  --capacity-mode reject `
  --capacity-observation absolute-and-demand-ratio
```

Train and compare the remaining seeds:

```powershell
python capacity_seed_convergence.py `
  --train `
  --seeds 17 27 37 `
  --episodes 12000 `
  --entropy-mode annealed `
  --near-capacities 2400 2300 2200 2100 2000 1900 1800 `
  --far-capacity 2400 `
  --failure-penalty -500 `
  --output-dir outputs_capacity_reject_seed_sweep `
  --include outputs_capacity_reject_annealed
```

Run the capacity stress test:

```powershell
python capacity_stress_test.py `
  --runs outputs_capacity_reject_annealed outputs_capacity_reject_seed_sweep `
  --training-seeds 7 17 27 37 `
  --output-dir outputs_capacity_stress_test
```

Run population-demand generalization:

```powershell
python capacity_demand_generalization.py `
  --runs outputs_capacity_reject_annealed outputs_capacity_reject_seed_sweep `
  --training-seeds 7 17 27 37 `
  --output-dir outputs_capacity_demand_generalization
```

Both evaluation scripts retain progress JSON files and can resume when the
same command is run again.

## Evaluation artifacts

- Capacity stress test:
  `outputs_capacity_stress_test/capacity_stress_test_20260918_140839_818924.json`
  and the matching PNG
- Demand generalization:
  `outputs_capacity_demand_generalization/capacity_demand_generalization_20260918_144715_397309.json`
  and the matching PNG

The JSON files, rather than values copied into prose, are the authoritative
numeric evaluation results.

## Software versions used for the recorded evaluation

- Python 3.13.10
- PyTorch 2.11.0+cu130
- NumPy 2.4.1
- Matplotlib 3.9.2

## Local full snapshot

The full local snapshot should include `.git`, all source and test scripts,
trained weights, diagnostics, configuration files, and evaluation outputs.
It is intended to preserve the complete experiment state even though large
artifacts are intentionally excluded from the GitHub commit.
