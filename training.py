"""Training entry point using the original project layout and naming.

All environment, return-target, advantage, and PPO computations match
the validated Codex implementation; only organization/names differ.
"""

"""From-scratch shared-actor, centralized-critic PPO implementation."""

from dataclasses import dataclass
from typing import Dict, Sequence
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import torch
from torch import nn

from common import (
    ADVANTAGE_EPSILON,
    ENTROPY_ANNEAL_EPISODES,
    ENTROPY_COEF,
    ENTROPY_FINAL,
    GAE_LAMBDA,
    GAMMA,
    HIDDEN_SIZE,
    LR_ACTOR,
    LR_CRITIC,
    MAX_GRAD_NORM,
    MAX_STEPS_EP,
    MINIBATCH_SIZE,
    N_TRAIN_AGENTS,
    PPO_CLIP,
    PPO_EPOCHS,
    ROLLOUT_EPISODES,
    SEED,
    TEAM_RETURN_GAMMA,
    TOTAL_EPISODES,
)
from torch.distributions import Categorical

import argparse
import json
import pickle
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from evac_env import EvacuationConfig, EvacuationEnv


class Actor(nn.Module):
    def __init__(self, input_dim: int = 5) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE), nn.Tanh(), nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 2), nn.Softmax(dim=-1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class Critic(nn.Module):
    def __init__(self, input_dim: int = 5) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE), nn.Tanh(), nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(), nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


@dataclass
class PPOConfig:
    observation_dim: int = 5
    critic_state_dim: int = 5
    actor_learning_rate: float = LR_ACTOR
    critic_learning_rate: float = LR_CRITIC
    gamma: float = GAMMA
    team_return_gamma: float = TEAM_RETURN_GAMMA
    gae_lambda: float = GAE_LAMBDA
    clip_epsilon: float = PPO_CLIP
    ppo_epochs: int = PPO_EPOCHS
    minibatch_size: int = MINIBATCH_SIZE
    rollout_episodes: int = ROLLOUT_EPISODES
    max_grad_norm: float = MAX_GRAD_NORM
    entropy_mode: str = "constant"
    entropy_coefficient: float = ENTROPY_COEF
    entropy_start: float = ENTROPY_COEF
    entropy_end: float = ENTROPY_FINAL
    entropy_anneal_episodes: int = ENTROPY_ANNEAL_EPISODES
    advantage_epsilon: float = ADVANTAGE_EPSILON
    advantage_reward: str = "team_mean_reward_to_go"


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE for one trajectory (terminal samples do not leak across rows)."""
    deltas = rewards + gamma * next_values * (1.0 - dones) - values
    advantages = np.empty_like(rewards, dtype=np.float32)
    gae = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        gae = deltas[index] + gamma * gae_lambda * (1.0 - dones[index]) * gae
        advantages[index] = gae
    return advantages, advantages + values


def _require_finite(name: str, value: torch.Tensor | np.ndarray | float) -> None:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if not bool(torch.isfinite(tensor).all()):
        bad = int((~torch.isfinite(tensor)).sum().item())
        raise FloatingPointError(f"Non-finite value detected in {name} ({bad} element(s))")


def masked_probabilities(probabilities: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
    """Remove physically unavailable actions and renormalize the policy."""
    if probabilities.shape != action_masks.shape:
        raise ValueError(
            f"Action mask shape {tuple(action_masks.shape)} does not match "
            f"probabilities {tuple(probabilities.shape)}"
        )
    masked = probabilities * action_masks
    totals = masked.sum(dim=-1, keepdim=True)
    if bool((totals <= 0.0).any()):
        raise RuntimeError("No available shelter edge for at least one decision")
    result = masked / totals
    _require_finite("masked action probabilities", result)
    return result


class MAPPOAgent:
    def __init__(self, config: PPOConfig, device: torch.device):
        self.config = config
        self.device = device
        self.actor = Actor(config.observation_dim).to(device)
        self.critic = Critic(config.critic_state_dim).to(device)
        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )
        self.update_count = 0

    @torch.no_grad()
    def act(
        self, observations: np.ndarray, action_masks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(action_masks, dtype=torch.float32, device=self.device)
        probabilities = masked_probabilities(self.actor(tensor), masks)
        _require_finite("action probabilities", probabilities)
        distribution = Categorical(probs=probabilities)
        actions = distribution.sample()
        return actions.cpu().numpy(), distribution.log_prob(actions).cpu().numpy(), probabilities.cpu().numpy()

    def entropy_coefficient(self, episode: int) -> float:
        c = self.config
        if c.entropy_mode == "constant":
            return c.entropy_coefficient
        if c.entropy_mode != "annealed":
            raise ValueError(f"Unknown entropy mode: {c.entropy_mode}")
        fraction = min(max(episode / float(c.entropy_anneal_episodes), 0.0), 1.0)
        return c.entropy_start + fraction * (c.entropy_end - c.entropy_start)

    def update(self, batch: Dict[str, np.ndarray], entropy_coef: float) -> Dict[str, float]:
        """Run PPO epochs; old log probabilities and normalized advantages stay fixed."""
        c = self.config
        obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device).detach()
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device).detach()
        action_masks = torch.as_tensor(
            batch["action_masks"], dtype=torch.float32, device=self.device
        ).detach()
        for name, value in (("observations", obs), ("states", states),
                            ("old_log_probability", old_log_probs), ("return", returns),
                            ("action mask", action_masks)):
            _require_finite(name, value)
        size = len(actions)
        if size == 0:
            raise ValueError("Cannot update PPO from an empty rollout")
        with torch.no_grad():
            advantages = (returns - self.critic(states)).detach()
            _require_finite("advantage before normalization", advantages)
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + c.advantage_epsilon
            )
            _require_finite("normalized advantage", advantages)

        sums = {key: 0.0 for key in (
            "actor_loss", "critic_loss", "policy_entropy", "mean_prob_near",
            "mean_prob_far", "approx_kl", "clip_fraction"
        )}
        sample_count = 0
        for _ in range(c.ppo_epochs):
            # One permutation per epoch means no duplicate or missing samples.
            permutation = torch.randperm(size, device=self.device)
            for start in range(0, size, c.minibatch_size):
                ids = permutation[start : start + c.minibatch_size]
                probabilities = masked_probabilities(self.actor(obs[ids]), action_masks[ids])
                distribution = Categorical(probs=probabilities)
                new_log_probs = distribution.log_prob(actions[ids])
                ratio = torch.exp(new_log_probs - old_log_probs[ids])
                mb_advantages = advantages[ids]
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - c.clip_epsilon, 1.0 + c.clip_epsilon) * mb_advantages
                entropy = distribution.entropy()
                actor_loss = -torch.minimum(surr1, surr2).mean() - entropy_coef * entropy.mean()
                for name, value in (("new_log_probability", new_log_probs),
                                    ("probability ratio", ratio), ("actor loss", actor_loss),
                                    ("policy entropy", entropy)):
                    _require_finite(name, value)
                self.optimizer_actor.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm)
                self.optimizer_actor.step()

                # Fresh critic forward pass for every minibatch in every epoch.
                predicted_values = self.critic(states[ids])
                critic_loss = 0.5 * (predicted_values - returns[ids]).pow(2).mean()
                _require_finite("critic prediction", predicted_values)
                _require_finite("critic loss", critic_loss)
                self.optimizer_critic.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
                self.optimizer_critic.step()

                n = len(ids)
                with torch.no_grad():
                    approx_kl = (old_log_probs[ids] - new_log_probs).mean()
                    clip_fraction = ((ratio - 1.0).abs() > c.clip_epsilon).float().mean()
                    _require_finite("approximate KL", approx_kl)
                    _require_finite("clip fraction", clip_fraction)
                    values = {
                        "actor_loss": actor_loss, "critic_loss": critic_loss,
                        "policy_entropy": entropy.mean(),
                        "mean_prob_near": probabilities[:, 0].mean(),
                        "mean_prob_far": probabilities[:, 1].mean(),
                        "approx_kl": approx_kl, "clip_fraction": clip_fraction,
                    }
                    for key, value in values.items():
                        sums[key] += float(value.item()) * n
                sample_count += n

        self.update_count += 1
        result = {key: value / sample_count for key, value in sums.items()}
        result.update({
            "sampled_far_fraction": float((actions == 1).float().mean().item()),
            "entropy_coefficient": float(entropy_coef),
            "transition_count": size,
            "update": self.update_count,
        })
        for name, value in result.items():
            _require_finite(name, value)
        return result


# Compatibility aliases allow diagnostics written for the stable Codex layout
# to load checkpoints/classes without changing the computation.
CentralizedCritic = Critic
MAPPO = MAPPOAgent
generalized_advantage_estimate = compute_gae


def ppo_update(agent_system: MAPPOAgent, batch: dict, entropy_coef: float) -> dict:
    """Legacy-style function name delegating to the validated PPO update."""
    return agent_system.update(batch, entropy_coef)

class RolloutBuffer:
    """Accumulate complete episodes and release only a full PPO rollout."""

    def __init__(self, episodes_per_update: int = 4):
        if episodes_per_update <= 0:
            raise ValueError("episodes_per_update must be positive")
        self.episodes_per_update = episodes_per_update
        self._episodes: list[dict[str, np.ndarray]] = []

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    @property
    def ready(self) -> bool:
        return self.episode_count == self.episodes_per_update

    def add(self, episode: dict[str, np.ndarray]) -> None:
        if self.ready:
            raise RuntimeError("RolloutBuffer is full; consume it before adding")
        self._episodes.append(episode)

    def consume(self) -> dict[str, np.ndarray]:
        if not self.ready:
            raise RuntimeError(
                f"Need {self.episodes_per_update} complete episodes, have {self.episode_count}"
            )
        keys = self._episodes[0].keys()
        combined = {key: np.concatenate([item[key] for item in self._episodes]) for key in keys}
        self._episodes.clear()
        return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
    parser.add_argument("--num-agents", type=int, default=N_TRAIN_AGENTS)
    parser.add_argument(
        "--num-agent-candidates", "--demand-levels", type=int, nargs="+",
        dest="num_agent_candidates",
        help=(
            "episode demand levels used in balanced mixed-demand training; "
            "for example: 1000 2000 3000 4000 5000"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_EP)
    parser.add_argument("--road-width", type=float, default=5.0)
    parser.add_argument(
        "--team-return-gamma",
        type=float,
        default=TEAM_RETURN_GAMMA,
        help=(
            "discount used only when summing future team rewards for each "
            "one-shot shelter decision (default: 1.0, no per-second discount)"
        ),
    )
    parser.add_argument("--entropy-mode", choices=("constant", "annealed"), default="constant")
    parser.add_argument(
        "--entropy-coefficient", type=float, default=ENTROPY_COEF,
        help="fixed entropy coefficient used when --entropy-mode constant",
    )
    parser.add_argument(
        "--entropy-start", type=float, default=ENTROPY_COEF,
        help="initial entropy coefficient used when --entropy-mode annealed",
    )
    parser.add_argument(
        "--entropy-end", type=float, default=ENTROPY_FINAL,
        help="final entropy coefficient used when --entropy-mode annealed",
    )
    parser.add_argument(
        "--entropy-anneal-episodes", type=int, default=ENTROPY_ANNEAL_EPISODES,
        help="episode at which linear entropy annealing reaches --entropy-end",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--torch-threads", type=int, default=0,
        help=(
            "PyTorch intra/inter-op CPU threads in the main process; "
            "0 keeps the PyTorch default"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--log-every-updates", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument(
        "--parallel-rollout-workers", type=int, default=1,
        help=(
            "number of CPU worker processes used to collect the independent "
            "episodes in each PPO rollout; 1 preserves serial collection"
        ),
    )
    parser.add_argument(
        "--validation-every-updates", type=int, default=0,
        help=(
            "evaluate on fixed validation seeds every N PPO updates; "
            "0 disables validation/best-model selection"
        ),
    )
    parser.add_argument(
        "--checkpoint-every-updates", type=int, default=0,
        help=(
            "save latest_checkpoint.pt every N PPO updates even before validation; "
            "0 saves only at validation/end"
        ),
    )
    parser.add_argument(
        "--validation-start-episode", type=int, default=2500,
        help="do not run periodic validation before this completed episode",
    )
    parser.add_argument(
        "--validation-seeds", type=int, nargs="+",
        default=[10000, 10001, 10002, 10003, 10004],
        help="fixed seeds used only for model selection, never for PPO updates",
    )
    parser.add_argument(
        "--joint-fixed-far-probabilities", type=float, nargs="+",
        default=[value / 100.0 for value in range(24, 37)],
        help=(
            "fixed-policy probabilities swept on validation seeds to define "
            "the return/arrival baseline for best_joint_actor.pt"
        ),
    )
    parser.add_argument(
        "--skip-joint-fixed-baseline", action="store_true",
        help=(
            "skip the fixed-probability validation sweep; useful when selecting "
            "best_safe_actor.pt across many demand/capacity conditions"
        ),
    )
    parser.add_argument(
        "--robust-joint-min-return-margin", type=float, default=0.5,
        help="minimum Actor return improvement over best fixed for robust saving",
    )
    parser.add_argument(
        "--robust-joint-min-arrival-margin", type=float, default=1.0,
        help="minimum arrival-time reduction in seconds for robust saving",
    )
    parser.add_argument(
        "--early-stopping", action="store_true",
        help=(
            "stop after the predeclared convergence/safety/validation plateau "
            "criteria are satisfied"
        ),
    )
    parser.add_argument("--early-stop-min-episode", type=int, default=4000)
    parser.add_argument(
        "--early-stop-window", type=int, default=500,
        help="size of each of the previous/recent training windows",
    )
    parser.add_argument(
        "--early-stop-return-tolerance", type=float, default=0.5,
    )
    parser.add_argument(
        "--early-stop-arrival-tolerance", type=float, default=2.0,
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=3,
        help="validation checks without material robust improvement",
    )
    parser.add_argument(
        "--early-stop-min-validation-seeds", type=int, default=20,
    )
    parser.add_argument(
        "--early-stop-return-improvement", type=float, default=0.1,
        help="validation return increase that resets plateau patience",
    )
    parser.add_argument(
        "--early-stop-arrival-improvement", type=float, default=0.5,
        help="validation arrival-time decrease that resets plateau patience",
    )
    parser.add_argument(
        "--common-checkpoint-episode", type=int, default=4000,
        help="save actor_epNNNN.pt at this common comparison episode; 0 disables",
    )
    parser.add_argument(
        "--resume", type=Path,
        help="resume exactly from a latest_checkpoint.pt written by this script",
    )
    parser.add_argument(
        "--profile-runtime", action="store_true",
        help="print and save rollout/PPO wall-clock timing for each update",
    )
    parser.add_argument(
        "--blockage-probability", type=float, default=0.0,
        help="probability that an episode contains one random complete edge blockage",
    )
    parser.add_argument("--blockage-min-duration", type=int, default=30)
    parser.add_argument("--blockage-max-duration", type=int, default=180)
    parser.add_argument(
        "--near-capacities", type=int, nargs="+",
        help="near-shelter capacities sampled uniformly per episode",
    )
    parser.add_argument(
        "--far-capacity", type=int,
        help="fixed far-shelter capacity (default: unlimited/no capacity experiment)",
    )
    parser.add_argument(
        "--near-capacity-ratios", type=float, nargs="+",
        help=(
            "near capacity / N values for balanced mixed-demand training; "
            "requires --num-agent-candidates"
        ),
    )
    parser.add_argument(
        "--far-capacity-ratio", type=float, default=0.8,
        help="far capacity / N in mixed-demand training (default: 0.8)",
    )
    parser.add_argument(
        "--demand-observation-scale", type=int, default=5000,
        help="denominator for the Actor/Critic total-demand feature N/scale",
    )
    parser.add_argument(
        "--failure-penalty", type=float,
        help="penalty per agent not arrived by max_steps (default: -500 with capacities, else 0)",
    )
    parser.add_argument(
        "--capacity-mode", choices=("hard-mask", "reject"), default="hard-mask",
        help="full-shelter handling during capacity training",
    )
    parser.add_argument(
        "--capacity-observation",
        choices=("absolute", "demand-ratio", "absolute-and-demand-ratio"),
        default="absolute",
        help=(
            "capacity feature schema (demand-ratio: Actor 5/Critic 7; "
            "absolute-and-demand-ratio: Actor 7/Critic 9)"
        ),
    )
    return parser.parse_args()


EpisodeCondition = tuple[int, int, int]


def build_mixed_demand_conditions(
    demands: Sequence[int],
    near_capacity_ratios: Sequence[float],
    far_capacity_ratio: float,
) -> tuple[EpisodeCondition, ...]:
    """Return the Cartesian product (N, near capacity, far capacity)."""
    return tuple(
        (
            int(demand),
            int(round(float(near_ratio) * demand)),
            int(round(float(far_capacity_ratio) * demand)),
        )
        for demand in demands
        for near_ratio in near_capacity_ratios
    )


def balanced_episode_condition(
    conditions: Sequence[EpisodeCondition], episode_index: int, seed: int
) -> EpisodeCondition:
    """Use every condition once per shuffled cycle, reproducibly on resume."""
    if not conditions:
        raise ValueError("At least one mixed-demand condition is required")
    cycle, position = divmod(int(episode_index), len(conditions))
    cycle_rng = np.random.default_rng(np.random.SeedSequence([seed, cycle, 9173]))
    order = cycle_rng.permutation(len(conditions))
    return tuple(conditions[int(order[position])])


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(name)


def make_new_run_directory(parent: Path, mode: str, seed: int) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    stem = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{mode}_seed{seed}"
    run_dir = parent / stem
    counter = 1
    while run_dir.exists():
        run_dir = parent / f"{stem}_{counter}"
        counter += 1
    run_dir.mkdir()
    return run_dir


def _atomic_torch_save(payload: object, path: Path) -> None:
    """Replace one checkpoint file without accumulating historical copies."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _validation_episode(
    config: EvacuationConfig,
    mappo_brain: MAPPOAgent,
    seed: int,
    fixed_far_probability: float | None = None,
    episode_condition: EpisodeCondition | None = None,
) -> dict:
    """Evaluate the Actor or one fixed policy without building a PPO rollout."""
    env = EvacuationEnv(config, seed=seed)
    env.reset(seed=seed, episode_condition=episode_condition)
    action_rng = np.random.default_rng(seed + 1_000_003)
    while not env.finished():
        ids, observations, _ = env.decision_batch()
        if len(ids):
            remaining = env.remaining_shelter_capacity()
            boundary_batch = np.any((remaining > 0) & (remaining < len(ids)))
            decision_groups = (
                [np.asarray([agent_id]) for agent_id in env.rng.permutation(ids)]
                if boundary_batch else [ids]
            )
            for group_ids in decision_groups:
                if boundary_batch:
                    observations, _ = env.observations_for(group_ids)
                mask = env.action_mask()
                masks = np.repeat(mask[None, :], len(group_ids), axis=0)
                if fixed_far_probability is None:
                    with torch.no_grad():
                        observation_tensor = torch.as_tensor(
                            observations, dtype=torch.float32,
                            device=mappo_brain.device,
                        )
                        mask_tensor = torch.as_tensor(
                            masks, dtype=torch.float32, device=mappo_brain.device
                        )
                        probabilities = masked_probabilities(
                            mappo_brain.actor(observation_tensor), mask_tensor
                        )[:, 1].cpu().numpy()
                else:
                    base = np.asarray(
                        [1.0 - fixed_far_probability, fixed_far_probability],
                        dtype=np.float64,
                    )
                    effective = base * mask
                    if effective.sum() <= 0.0:
                        effective = mask.astype(np.float64)
                    if effective.sum() <= 0.0:
                        raise RuntimeError("No available shelter during validation")
                    probabilities = np.full(
                        len(group_ids), effective[1] / effective.sum()
                    )
                actions = (
                    action_rng.random(len(group_ids)) < probabilities
                ).astype(np.int8)
                env.commit(group_ids, actions)
        env.advance(mappo_brain.config.gamma)
    return env.finalize()


def evaluate_fixed_validation_seeds(
    config: EvacuationConfig,
    mappo_brain: MAPPOAgent,
    seeds: list[int],
    fixed_far_probability: float | None = None,
    conditions: Sequence[EpisodeCondition] | None = None,
) -> dict[str, object]:
    """Evaluate reproducibly and restore every RNG used by training afterward."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = mappo_brain.actor.training
    rows = []
    try:
        mappo_brain.actor.eval()
        validation_conditions = tuple(conditions) if conditions else (None,)
        for condition in validation_conditions:
            for seed in seeds:
                # Reusing these seeds at every checkpoint makes changes attributable
                # to the Actor rather than to different environment/action draws.
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                row = _validation_episode(
                    config, mappo_brain, seed, fixed_far_probability,
                    episode_condition=condition,
                )
                row["condition"] = condition
                rows.append(row)
    finally:
        if was_training:
            mappo_brain.actor.train()
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    condition_summaries = []
    for condition in (tuple(conditions) if conditions else (None,)):
        condition_rows = [row for row in rows if row["condition"] == condition]
        condition_summaries.append({
            "num_agents": int(condition[0]) if condition else config.num_agents,
            "near_shelter_capacity": (
                int(condition[1]) if condition
                else int(condition_rows[0]["near_shelter_capacity"])
            ),
            "far_shelter_capacity": (
                int(condition[2]) if condition
                else int(condition_rows[0]["far_shelter_capacity"])
            ),
            "mean_episode_return_per_agent": float(np.mean([
                row["mean_agent_reward"] for row in condition_rows
            ])),
            "mean_arrival_time_with_timeouts": float(np.mean([
                row["mean_arrival_time_all_agents"] for row in condition_rows
            ])),
            "sampled_far_fraction": float(np.mean([
                row["far_fraction"] for row in condition_rows
            ])),
            "mean_failure_count": float(np.mean([
                row["failure_count"] for row in condition_rows
            ])),
            "max_failure_count": int(max(
                row["failure_count"] for row in condition_rows
            )),
        })

    return {
        "mean_episode_return_per_agent": float(np.mean([
            row["mean_agent_reward"] for row in rows
        ])),
        "mean_arrival_time_with_timeouts": float(np.mean([
            row["mean_arrival_time_all_agents"] for row in rows
        ])),
        "sampled_far_fraction": float(np.mean([
            row["far_fraction"] for row in rows
        ])),
        "failure_count": float(np.mean([
            row["failure_count"] for row in rows
        ])),
        "max_failure_count": int(max(row["failure_count"] for row in rows)),
        "condition_summaries": condition_summaries,
    }


def _checkpoint_payload(
    episode_completed: int,
    env: EvacuationEnv,
    mappo_brain: MAPPOAgent,
    diagnostics: dict,
    best_validation_return: float,
    best_joint_validation_return: float,
    best_joint_validation_arrival: float,
    best_robust_joint_validation_return: float,
    best_robust_joint_validation_arrival: float,
) -> dict:
    return {
        "episode_completed": episode_completed,
        "actor_state_dict": mappo_brain.actor.state_dict(),
        "critic_state_dict": mappo_brain.critic.state_dict(),
        "actor_optimizer_state_dict": mappo_brain.optimizer_actor.state_dict(),
        "critic_optimizer_state_dict": mappo_brain.optimizer_critic.state_dict(),
        "update_count": mappo_brain.update_count,
        "environment_rng_state": env.rng.bit_generator.state,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "diagnostics": diagnostics,
        "best_validation_return": best_validation_return,
        "best_joint_validation_return": best_joint_validation_return,
        "best_joint_validation_arrival": best_joint_validation_arrival,
        "best_robust_joint_validation_return": best_robust_joint_validation_return,
        "best_robust_joint_validation_arrival": best_robust_joint_validation_arrival,
    }


def collect_episode(
    env: EvacuationEnv,
    mappo_brain: MAPPOAgent,
    reset_seed: int | None = None,
    episode_condition: EpisodeCondition | None = None,
) -> tuple[dict, dict]:
    observations, states, actions, action_masks, log_probs, probabilities, agent_ids = [], [], [], [], [], [], []
    env.reset(seed=reset_seed, episode_condition=episode_condition)
    while not env.finished():
        ids, obs, global_states = env.decision_batch()
        if len(ids):
            remaining = env.remaining_shelter_capacity()
            # A vectorized batch is exact unless a shelter has fewer free slots
            # than this departure group. Only that boundary batch is processed
            # in random reservation order, avoiding systematic agent-ID priority
            # while keeping each stored observation/log-probability consistent.
            boundary_batch = np.any((remaining > 0) & (remaining < len(ids)))
            decision_groups = (
                [np.asarray([agent_id]) for agent_id in env.rng.permutation(ids)]
                if boundary_batch else [ids]
            )
            for group_ids in decision_groups:
                if boundary_batch:
                    obs, global_states = env.observations_for(group_ids)
                mask = env.action_mask()
                masks = np.repeat(mask[None, :], len(group_ids), axis=0)
                sampled_actions, sampled_log_probs, action_probabilities = mappo_brain.act(obs, masks)
                env.commit(group_ids, sampled_actions)
                observations.append(obs)
                states.append(global_states)
                actions.append(sampled_actions)
                action_masks.append(masks)
                log_probs.append(sampled_log_probs)
                probabilities.append(action_probabilities)
                agent_ids.append(group_ids)
        env.advance(mappo_brain.config.gamma)
    metrics = env.finalize()
    metrics["simulation_steps"] = int(env.current_step)

    ids = np.concatenate(agent_ids)
    order = np.argsort(ids)
    obs = np.concatenate(observations)[order]
    state = np.concatenate(states)[order]
    action = np.concatenate(actions)[order]
    masks = np.concatenate(action_masks)[order]
    old_log_prob = np.concatenate(log_probs)[order]
    probs = np.concatenate(probabilities)[order]
    decision_steps = env.start_steps[ids][order]
    team_reward_to_go = env.team_discounted_reward_to_go(
        mappo_brain.config.team_return_gamma
    )
    # The failure term is an episode outcome and is added without another gamma
    # discount. At step 900, gamma**900 would otherwise make it nearly invisible
    # to the one-shot decisions made near the beginning of the episode.
    if env.config.shelter_capacity_mode == "reject":
        # A rejected/timeout decision receives its own failure penalty. Averaged
        # across agents this is the same -penalty * failure_fraction objective,
        # but direct attribution gives the one-shot Actor a useful learning signal.
        failed = np.asarray(metrics["per_agent_failed"], dtype=np.float32)[ids][order]
        discounted_return = (
            team_reward_to_go[decision_steps]
            + env.config.failure_penalty * failed
        )
    else:
        discounted_return = team_reward_to_go[decision_steps] + metrics["team_failure_penalty"]

    # Every agent makes exactly one decision. Its terminal macro-transition uses
    # the system-wide mean reward-to-go from its departure step. This supplies the
    # cross-agent congestion externality missing from an individual return. With
    # next_value=0 and done=1, GAE equals team_reward_to_go - V; lambda has no
    # numerical effect and samples never bootstrap across agents or episodes.
    with torch.no_grad():
        values = mappo_brain.critic(
            torch.as_tensor(state, dtype=torch.float32, device=mappo_brain.device)
        ).cpu().numpy()
    _, return_targets = generalized_advantage_estimate(
        discounted_return, values, np.zeros_like(values), np.ones_like(values),
        mappo_brain.config.gamma, mappo_brain.config.gae_lambda,
    )
    batch = {
        "observations": obs,
        "states": state,
        "actions": action,
        "action_masks": masks,
        "log_probs": old_log_prob,
        "returns": return_targets,
    }
    metrics["sampled_far_fraction"] = metrics.pop("far_fraction")
    metrics["mean_prob_near"] = float(probs[:, 0].mean())
    metrics["mean_prob_far"] = float(probs[:, 1].mean())
    metrics["policy_entropy"] = float((-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)).mean())
    # The mappo_brain uses a team reward-to-go target, which is distinct from the
    # raw per-agent episode return reported by the environment.
    metrics["mean_training_return_target"] = float(np.mean(return_targets))
    # Stable diagnostics names matching the reported quantities.
    metrics["mean_episode_return_per_agent"] = metrics["mean_agent_reward"]
    metrics["arrived_only_mean_arrival_time"] = metrics["mean_arrival_time_arrived_only"]
    metrics["mean_arrival_time_with_timeouts"] = metrics["mean_arrival_time_all_agents"]
    metrics["arrived_count"] = metrics["arrival_count"]
    metrics["total_count"] = env.num_agents
    return batch, metrics


_ROLLOUT_WORKER_ENV: EvacuationEnv | None = None
_ROLLOUT_WORKER_AGENT: MAPPOAgent | None = None


def _initialize_rollout_worker(
    environment_config: dict,
    ppo_config: dict,
) -> None:
    """Create one persistent CPU environment/Actor per rollout worker."""
    global _ROLLOUT_WORKER_ENV, _ROLLOUT_WORKER_AGENT
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _ROLLOUT_WORKER_ENV = EvacuationEnv(
        EvacuationConfig(**environment_config), seed=0
    )
    _ROLLOUT_WORKER_AGENT = MAPPOAgent(
        PPOConfig(**ppo_config), torch.device("cpu")
    )


def _collect_parallel_episode(
    payload: tuple[dict, dict, int, EpisodeCondition | None]
) -> tuple[dict, dict]:
    """Load the current policy/value weights and collect one seeded episode."""
    actor_state, critic_state, episode_seed, episode_condition = payload
    if _ROLLOUT_WORKER_ENV is None or _ROLLOUT_WORKER_AGENT is None:
        raise RuntimeError("Rollout worker was not initialized")
    _ROLLOUT_WORKER_AGENT.actor.load_state_dict(actor_state)
    _ROLLOUT_WORKER_AGENT.critic.load_state_dict(critic_state)
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)
    return collect_episode(
        _ROLLOUT_WORKER_ENV,
        _ROLLOUT_WORKER_AGENT,
        reset_seed=episode_seed,
        episode_condition=episode_condition,
    )


def _parallel_episode_seed(base_seed: int, episode_index: int) -> int:
    """Deterministic per-episode seed, stable across checkpoint resume."""
    modulus = 2**31 - 1
    return int(
        (base_seed * 1_000_003 + (episode_index + 1) * 97_409) % modulus
    )


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _episode_rollout_stream(
    start_episode: int,
    end_episode: int,
    env: EvacuationEnv,
    agent: MAPPOAgent,
    executor: ProcessPoolExecutor | None,
    base_seed: int,
    conditions: Sequence[EpisodeCondition] = (),
):
    """Yield episodes serially or in policy-frozen rollout-sized groups."""
    if executor is None:
        for episode_index in range(start_episode, end_episode):
            started = time.perf_counter()
            episode_condition = (
                balanced_episode_condition(conditions, episode_index, base_seed)
                if conditions else None
            )
            batch, metrics = collect_episode(
                env, agent, episode_condition=episode_condition
            )
            yield episode_index, batch, metrics, time.perf_counter() - started
        return

    rollout_size = agent.config.rollout_episodes
    for group_start in range(start_episode, end_episode, rollout_size):
        indices = list(
            range(group_start, min(group_start + rollout_size, end_episode))
        )
        actor_state = _cpu_state_dict(agent.actor)
        critic_state = _cpu_state_dict(agent.critic)
        payloads = [
            (
                actor_state,
                critic_state,
                _parallel_episode_seed(base_seed, episode_index),
                (
                    balanced_episode_condition(
                        conditions, episode_index, base_seed
                    )
                    if conditions else None
                ),
            )
            for episode_index in indices
        ]
        started = time.perf_counter()
        results = list(executor.map(_collect_parallel_episode, payloads))
        group_seconds = time.perf_counter() - started
        # Attribute equal shares only for per-episode accounting; the sum is
        # the measured wall time for the complete parallel rollout group.
        seconds_per_result = group_seconds / len(results)
        for episode_index, (batch, metrics) in zip(indices, results):
            yield episode_index, batch, metrics, seconds_per_result


def _append(mapping: dict[str, list], record: dict, keys: tuple[str, ...]) -> None:
    for key in keys:
        mapping[key].append(record[key])


def _training_stability(
    diagnostics: dict,
    window: int,
    return_tolerance: float,
    arrival_tolerance: float,
) -> dict[str, float | bool]:
    """Compare the latest two non-overlapping training windows."""
    metrics = diagnostics["episode_metrics"]
    returns = np.asarray(metrics["mean_episode_return_per_agent"], dtype=float)
    arrivals = np.asarray(
        metrics["mean_arrival_time_with_timeouts"], dtype=float
    )
    if len(returns) < 2 * window:
        return {
            "training_stable": False,
            "return_change": float("nan"),
            "arrival_change": float("nan"),
        }
    previous = slice(len(returns) - 2 * window, len(returns) - window)
    recent = slice(len(returns) - window, len(returns))
    return_change = float(returns[recent].mean() - returns[previous].mean())
    arrival_change = float(
        arrivals[recent].mean() - arrivals[previous].mean()
    )
    return {
        "training_stable": (
            abs(return_change) <= return_tolerance
            and abs(arrival_change) <= arrival_tolerance
        ),
        "return_change": return_change,
        "arrival_change": arrival_change,
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.log_every_updates <= 0:
        raise ValueError("--log-every-updates must be positive")
    if args.torch_threads < 0:
        raise ValueError("--torch-threads must be non-negative")
    if args.parallel_rollout_workers <= 0:
        raise ValueError("--parallel-rollout-workers must be positive")
    if args.parallel_rollout_workers > ROLLOUT_EPISODES:
        raise ValueError(
            "--parallel-rollout-workers cannot exceed the number of "
            f"episodes per PPO rollout ({ROLLOUT_EPISODES})"
        )
    if args.validation_every_updates < 0:
        raise ValueError("--validation-every-updates must be non-negative")
    if args.checkpoint_every_updates < 0:
        raise ValueError("--checkpoint-every-updates must be non-negative")
    if args.validation_start_episode < 0:
        raise ValueError("--validation-start-episode must be non-negative")
    if not args.validation_seeds:
        raise ValueError("--validation-seeds must not be empty")
    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        raise ValueError("--validation-seeds contains duplicates")
    if not args.joint_fixed_far_probabilities:
        raise ValueError("--joint-fixed-far-probabilities must not be empty")
    if any(
        probability < 0.0 or probability > 1.0
        for probability in args.joint_fixed_far_probabilities
    ):
        raise ValueError("--joint-fixed-far-probabilities must be in [0, 1]")
    if args.entropy_coefficient < 0.0:
        raise ValueError("--entropy-coefficient must be non-negative")
    if args.entropy_start < 0.0 or args.entropy_end < 0.0:
        raise ValueError("--entropy-start and --entropy-end must be non-negative")
    if args.entropy_anneal_episodes <= 0:
        raise ValueError("--entropy-anneal-episodes must be positive")
    if args.robust_joint_min_return_margin < 0.0:
        raise ValueError("--robust-joint-min-return-margin must be non-negative")
    if args.robust_joint_min_arrival_margin < 0.0:
        raise ValueError("--robust-joint-min-arrival-margin must be non-negative")
    if args.early_stop_min_episode < 0:
        raise ValueError("--early-stop-min-episode must be non-negative")
    if args.early_stop_window <= 0:
        raise ValueError("--early-stop-window must be positive")
    if args.early_stop_return_tolerance < 0.0:
        raise ValueError("--early-stop-return-tolerance must be non-negative")
    if args.early_stop_arrival_tolerance < 0.0:
        raise ValueError("--early-stop-arrival-tolerance must be non-negative")
    if args.early_stop_patience <= 0:
        raise ValueError("--early-stop-patience must be positive")
    if args.early_stop_min_validation_seeds <= 0:
        raise ValueError("--early-stop-min-validation-seeds must be positive")
    if args.early_stop_return_improvement < 0.0:
        raise ValueError("--early-stop-return-improvement must be non-negative")
    if args.early_stop_arrival_improvement < 0.0:
        raise ValueError("--early-stop-arrival-improvement must be non-negative")
    if args.common_checkpoint_episode < 0:
        raise ValueError("--common-checkpoint-episode must be non-negative")
    if args.early_stopping:
        if args.validation_every_updates <= 0:
            raise ValueError(
                "--early-stopping requires --validation-every-updates > 0"
            )
        if args.episodes < args.early_stop_min_episode:
            raise ValueError(
                "--episodes must be at least --early-stop-min-episode"
            )
        if len(args.validation_seeds) < args.early_stop_min_validation_seeds:
            raise ValueError(
                "early stopping requires at least "
                f"{args.early_stop_min_validation_seeds} validation seeds"
            )
    if not 0.0 <= args.team_return_gamma <= 1.0:
        raise ValueError("--team-return-gamma must be in [0, 1]")
    if not 0.0 <= args.blockage_probability <= 1.0:
        raise ValueError("--blockage-probability must be in [0, 1]")
    if not 1 <= args.blockage_min_duration <= args.blockage_max_duration:
        raise ValueError("Blockage durations must satisfy 1 <= min <= max")
    if args.near_capacities is not None and any(value < 0 for value in args.near_capacities):
        raise ValueError("--near-capacities values must be non-negative")
    if args.far_capacity is not None and args.far_capacity < 0:
        raise ValueError("--far-capacity must be non-negative")
    mixed_demand = args.num_agent_candidates is not None
    if mixed_demand:
        if not args.num_agent_candidates or any(
            value <= 0 for value in args.num_agent_candidates
        ):
            raise ValueError("--num-agent-candidates values must be positive")
        if len(set(args.num_agent_candidates)) != len(args.num_agent_candidates):
            raise ValueError("--num-agent-candidates contains duplicates")
        if not args.near_capacity_ratios:
            raise ValueError(
                "--num-agent-candidates requires --near-capacity-ratios"
            )
        if args.near_capacities is not None or args.far_capacity is not None:
            raise ValueError(
                "Use capacity ratios, not --near-capacities/--far-capacity, "
                "with mixed-demand training"
            )
        if any(value < 0.0 for value in args.near_capacity_ratios):
            raise ValueError("--near-capacity-ratios values must be non-negative")
        if args.far_capacity_ratio < 0.0:
            raise ValueError("--far-capacity-ratio must be non-negative")
        if args.demand_observation_scale < max(args.num_agent_candidates):
            raise ValueError(
                "--demand-observation-scale must be at least the largest demand"
            )
    elif args.near_capacity_ratios is not None:
        raise ValueError(
            "--near-capacity-ratios requires --num-agent-candidates"
        )
    if args.demand_observation_scale <= 0:
        raise ValueError("--demand-observation-scale must be positive")
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        try:
            torch.set_num_interop_threads(args.torch_threads)
        except RuntimeError:
            pass
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    mixed_conditions = (
        build_mixed_demand_conditions(
            args.num_agent_candidates,
            args.near_capacity_ratios,
            args.far_capacity_ratio,
        )
        if mixed_demand else ()
    )
    if mixed_conditions and any(
        near_capacity + far_capacity < demand
        for demand, near_capacity, far_capacity in mixed_conditions
    ):
        raise ValueError(
            "Every mixed-demand near/far capacity combination must accommodate N"
        )
    capacity_enabled = (
        mixed_demand
        or args.near_capacities is not None
        or args.far_capacity is not None
    )
    selected_capacity_observation = args.capacity_observation
    if mixed_demand and selected_capacity_observation == "absolute":
        selected_capacity_observation = "absolute-and-demand-ratio"
        print(
            "Mixed-demand training: using absolute-and-demand-ratio capacity "
            "features plus normalized total demand.",
            flush=True,
        )
    if args.capacity_mode == "reject" and not capacity_enabled:
        raise ValueError("--capacity-mode reject requires shelter capacities")
    if args.capacity_observation != "absolute" and not capacity_enabled:
        raise ValueError("The selected --capacity-observation requires shelter capacities")
    if capacity_enabled and args.blockage_probability > 0.0:
        raise ValueError(
            "Capacity training currently requires --blockage-probability 0; "
            "train the two robustness mechanisms separately"
        )
    failure_penalty = (
        args.failure_penalty
        if args.failure_penalty is not None
        else (-500.0 if capacity_enabled else 0.0)
    )
    default_num_agents = mixed_conditions[0][0] if mixed_conditions else args.num_agents
    default_near_capacity = (
        mixed_conditions[0][1] if mixed_conditions else None
    )
    default_far_capacity = (
        mixed_conditions[0][2] if mixed_conditions else args.far_capacity
    )
    env_config = EvacuationConfig(
        num_agents=default_num_agents,
        max_steps=args.max_steps,
        road_width=args.road_width,
        random_blockage_probability=args.blockage_probability,
        random_blockage_min_duration=args.blockage_min_duration,
        random_blockage_max_duration=args.blockage_max_duration,
        near_shelter_capacity=default_near_capacity,
        far_shelter_capacity=default_far_capacity,
        near_shelter_capacity_candidates=(
            () if mixed_demand else tuple(args.near_capacities or ())
        ),
        failure_penalty=failure_penalty,
        include_capacity_observation=True,
        include_shelter_capacity_observation=capacity_enabled,
        shelter_capacity_mode=args.capacity_mode.replace("-", "_"),
        observation_schema={
            "absolute": "legacy",
            "demand-ratio": "capacity_ratio",
            "absolute-and-demand-ratio": "capacity_absolute_and_ratio",
        }[selected_capacity_observation],
        include_total_demand_observation=mixed_demand,
        total_demand_observation_scale=args.demand_observation_scale,
    )
    if selected_capacity_observation == "demand-ratio":
        actor_input_dim, critic_input_dim = 5, 7
    elif selected_capacity_observation == "absolute-and-demand-ratio":
        actor_input_dim, critic_input_dim = 7, 9
    else:
        actor_input_dim = critic_input_dim = 7 if capacity_enabled else 5
    if mixed_demand:
        actor_input_dim += 1
        critic_input_dim += 1
    ppo_config = PPOConfig(
        team_return_gamma=args.team_return_gamma,
        entropy_mode=args.entropy_mode,
        entropy_coefficient=args.entropy_coefficient,
        entropy_start=args.entropy_start,
        entropy_end=args.entropy_end,
        entropy_anneal_episodes=args.entropy_anneal_episodes,
        minibatch_size=args.minibatch_size,
        observation_dim=actor_input_dim,
        critic_state_dim=critic_input_dim,
    )
    env = EvacuationEnv(env_config, seed=args.seed)
    mappo_brain = MAPPOAgent(ppo_config, device)
    buffer = RolloutBuffer(ppo_config.rollout_episodes)
    run_dir = (
        args.resume.resolve().parent
        if args.resume is not None
        else make_new_run_directory(args.output_dir, args.entropy_mode, args.seed)
    )

    episode_keys = (
        "mean_episode_return_per_agent", "arrived_only_mean_arrival_time",
        "mean_arrival_time_with_timeouts", "arrived_count", "total_count", "failure_count",
        "sampled_far_fraction", "mean_prob_near", "mean_prob_far", "policy_entropy",
        "mean_training_return_target",
        "near_shelter_capacity", "far_shelter_capacity",
        "near_reserved_count", "far_reserved_count", "team_failure_penalty",
        "capacity_rejection_count", "timeout_count",
        "simulation_steps",
        "blockage_edge", "blockage_start_step", "blockage_end_step",
    )
    update_keys = (
        "actor_loss", "critic_loss", "policy_entropy", "entropy_coefficient",
        "mean_prob_near", "mean_prob_far", "sampled_far_fraction", "approx_kl",
        "clip_fraction", "transition_count",
    )
    diagnostics = {
        "environment_config": asdict(env_config),
        "ppo_config": asdict(ppo_config),
        "seed": args.seed,
        "entropy_mode": args.entropy_mode,
        "device": str(device),
        "episode": [],
        "episode_metrics": {key: [] for key in episode_keys},
        "update": [],
        "update_episode": [],
        "update_metrics": {key: [] for key in update_keys},
        "runtime": {
            "rollout_seconds_per_update": [],
            "ppo_update_seconds": [],
            "elapsed_seconds": [],
            "simulation_steps_per_update": [],
            "transitions_per_update": [],
        },
        "validation": [],
        "mixed_demand_conditions": [list(condition) for condition in mixed_conditions],
        "safe_model_selection": {
            "best_key": None,
            "best_episode": None,
            "checks": [],
        },
        "early_stopping": {
            "enabled": bool(args.early_stopping),
            "min_episode": int(args.early_stop_min_episode),
            "window": int(args.early_stop_window),
            "return_tolerance": float(args.early_stop_return_tolerance),
            "arrival_tolerance": float(args.early_stop_arrival_tolerance),
            "patience": int(args.early_stop_patience),
            "min_validation_seeds": int(
                args.early_stop_min_validation_seeds
            ),
            "return_improvement": float(
                args.early_stop_return_improvement
            ),
            "arrival_improvement": float(
                args.early_stop_arrival_improvement
            ),
            "plateau_checks": 0,
            "best_eligible_return": None,
            "best_eligible_arrival": None,
            "checks": [],
            "stopped": False,
            "stop_episode": None,
            "reason": None,
        },
    }
    start_episode = 0
    best_validation_return = -float("inf")
    best_joint_validation_return = -float("inf")
    best_joint_validation_arrival = float("inf")
    best_robust_joint_validation_return = -float("inf")
    best_robust_joint_validation_arrival = float("inf")
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.resume, map_location=device)
        mappo_brain.actor.load_state_dict(checkpoint["actor_state_dict"])
        mappo_brain.critic.load_state_dict(checkpoint["critic_state_dict"])
        mappo_brain.optimizer_actor.load_state_dict(
            checkpoint["actor_optimizer_state_dict"]
        )
        mappo_brain.optimizer_critic.load_state_dict(
            checkpoint["critic_optimizer_state_dict"]
        )
        mappo_brain.update_count = int(checkpoint["update_count"])
        env.rng.bit_generator.state = checkpoint["environment_rng_state"]
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all([
                state.cpu() for state in checkpoint["cuda_rng_states"]
            ])
        diagnostics = checkpoint["diagnostics"]
        diagnostics.setdefault("validation", [])
        diagnostics.setdefault("safe_model_selection", {
            "best_key": None,
            "best_episode": None,
            "checks": [],
        })
        start_episode = int(checkpoint["episode_completed"])
        best_validation_return = float(
            checkpoint.get("best_validation_return", -float("inf"))
        )
        best_joint_validation_return = float(
            checkpoint.get("best_joint_validation_return", -float("inf"))
        )
        best_joint_validation_arrival = float(
            checkpoint.get("best_joint_validation_arrival", float("inf"))
        )
        best_robust_joint_validation_return = float(
            checkpoint.get("best_robust_joint_validation_return", -float("inf"))
        )
        best_robust_joint_validation_arrival = float(
            checkpoint.get("best_robust_joint_validation_arrival", float("inf"))
        )
        if args.episodes <= start_episode:
            raise ValueError(
                f"--episodes ({args.episodes}) must exceed the completed checkpoint "
                f"episode ({start_episode})"
            )
        print(
            f"Resuming {args.resume.resolve()} from episode {start_episode}, "
            f"update {mappo_brain.update_count}",
            flush=True,
        )

    early_stop_state = diagnostics.setdefault("early_stopping", {})
    early_stop_state.update({
        "enabled": bool(args.early_stopping),
        "min_episode": int(args.early_stop_min_episode),
        "window": int(args.early_stop_window),
        "return_tolerance": float(args.early_stop_return_tolerance),
        "arrival_tolerance": float(args.early_stop_arrival_tolerance),
        "patience": int(args.early_stop_patience),
        "min_validation_seeds": int(args.early_stop_min_validation_seeds),
        "return_improvement": float(args.early_stop_return_improvement),
        "arrival_improvement": float(args.early_stop_arrival_improvement),
    })
    early_stop_state.setdefault("plateau_checks", 0)
    early_stop_state.setdefault("best_eligible_return", None)
    early_stop_state.setdefault("best_eligible_arrival", None)
    early_stop_state.setdefault("checks", [])
    # A resumed run is a new opportunity to satisfy the same predeclared rule.
    early_stop_state["stopped"] = False
    early_stop_state["stop_episode"] = None
    early_stop_state["reason"] = None

    joint_fixed_baseline = None
    if args.validation_every_updates > 0 and not args.skip_joint_fixed_baseline:
        fixed_sweep = []
        for probability in args.joint_fixed_far_probabilities:
            fixed_result = evaluate_fixed_validation_seeds(
                env_config, mappo_brain, args.validation_seeds,
                fixed_far_probability=float(probability),
                conditions=mixed_conditions,
            )
            fixed_result["nominal_far_probability"] = float(probability)
            fixed_sweep.append(fixed_result)
        joint_fixed_baseline = max(
            fixed_sweep,
            key=lambda row: row["mean_episode_return_per_agent"],
        )
        diagnostics["joint_fixed_validation_sweep"] = fixed_sweep
        diagnostics["joint_fixed_baseline"] = joint_fixed_baseline
        print(
            "joint fixed baseline "
            f"p_far={joint_fixed_baseline['nominal_far_probability']:.3f} "
            f"return={joint_fixed_baseline['mean_episode_return_per_agent']:.3f} "
            f"arrival_time={joint_fixed_baseline['mean_arrival_time_with_timeouts']:.1f}s "
            f"failures={joint_fixed_baseline['failure_count']:.2f}",
            flush=True,
        )

        # Older checkpoints do not contain joint-selection state. Evaluate the
        # resumed latest Actor once so it can immediately become a joint winner.
        if args.resume is not None:
            resumed_validation = evaluate_fixed_validation_seeds(
                env_config, mappo_brain, args.validation_seeds,
                conditions=mixed_conditions,
            )
            resumed_eligible = (
                resumed_validation["failure_count"]
                <= joint_fixed_baseline["failure_count"]
                and resumed_validation["mean_episode_return_per_agent"]
                > joint_fixed_baseline["mean_episode_return_per_agent"]
                and resumed_validation["mean_arrival_time_with_timeouts"]
                < joint_fixed_baseline["mean_arrival_time_with_timeouts"]
            )
            resumed_joint_is_best = resumed_eligible and (
                resumed_validation["mean_episode_return_per_agent"]
                > best_joint_validation_return
                or (
                    resumed_validation["mean_episode_return_per_agent"]
                    == best_joint_validation_return
                    and resumed_validation["mean_arrival_time_with_timeouts"]
                    < best_joint_validation_arrival
                )
            )
            resumed_return_margin = (
                resumed_validation["mean_episode_return_per_agent"]
                - joint_fixed_baseline["mean_episode_return_per_agent"]
            )
            resumed_arrival_margin = (
                joint_fixed_baseline["mean_arrival_time_with_timeouts"]
                - resumed_validation["mean_arrival_time_with_timeouts"]
            )
            resumed_robust_eligible = (
                resumed_validation["failure_count"]
                <= joint_fixed_baseline["failure_count"]
                and resumed_return_margin >= args.robust_joint_min_return_margin
                and resumed_arrival_margin >= args.robust_joint_min_arrival_margin
            )
            resumed_robust_is_best = resumed_robust_eligible and (
                resumed_validation["mean_episode_return_per_agent"]
                > best_robust_joint_validation_return
                or (
                    resumed_validation["mean_episode_return_per_agent"]
                    == best_robust_joint_validation_return
                    and resumed_validation["mean_arrival_time_with_timeouts"]
                    < best_robust_joint_validation_arrival
                )
            )
            if resumed_joint_is_best:
                best_joint_validation_return = resumed_validation[
                    "mean_episode_return_per_agent"
                ]
                best_joint_validation_arrival = resumed_validation[
                    "mean_arrival_time_with_timeouts"
                ]
                _atomic_torch_save(
                    mappo_brain.actor.state_dict(), run_dir / "best_joint_actor.pt"
                )
            if resumed_robust_is_best:
                best_robust_joint_validation_return = resumed_validation[
                    "mean_episode_return_per_agent"
                ]
                best_robust_joint_validation_arrival = resumed_validation[
                    "mean_arrival_time_with_timeouts"
                ]
                _atomic_torch_save(
                    mappo_brain.actor.state_dict(),
                    run_dir / "best_robust_joint_actor.pt",
                )
            diagnostics.setdefault("joint_resume_validation", []).append({
                **resumed_validation,
                "episode": start_episode,
                "eligible": resumed_eligible,
                "joint_is_best": resumed_joint_is_best,
                "return_margin_over_fixed": resumed_return_margin,
                "arrival_margin_over_fixed": resumed_arrival_margin,
                "robust_joint_eligible": resumed_robust_eligible,
                "robust_joint_is_best": resumed_robust_is_best,
            })
    recent_metrics: list[dict] = []
    run_started = time.perf_counter()
    rollout_seconds_total = 0.0
    ppo_seconds_total = 0.0
    rollout_seconds_since_update = 0.0
    simulation_steps_total = 0
    simulation_steps_since_update = 0
    transitions_total = 0
    stop_training = False
    stop_reason = None
    completed_episode_count = start_episode
    rollout_executor = None
    if args.parallel_rollout_workers > 1:
        rollout_executor = ProcessPoolExecutor(
            max_workers=args.parallel_rollout_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_rollout_worker,
            initargs=(asdict(env_config), asdict(ppo_config)),
        )
        print(
            f"Parallel rollout enabled with "
            f"{args.parallel_rollout_workers} CPU workers",
            flush=True,
        )

    for episode_index, batch, metrics, rollout_seconds in _episode_rollout_stream(
        start_episode,
        args.episodes,
        env,
        mappo_brain,
        rollout_executor,
        args.seed,
        mixed_conditions,
    ):
        rollout_seconds_total += rollout_seconds
        rollout_seconds_since_update += rollout_seconds
        simulation_steps = int(metrics["simulation_steps"])
        simulation_steps_total += simulation_steps
        simulation_steps_since_update += simulation_steps
        transitions_total += len(batch["actions"])
        buffer.add(batch)
        recent_metrics.append(metrics)
        diagnostics["episode"].append(episode_index + 1)
        _append(diagnostics["episode_metrics"], metrics, episode_keys)
        if not buffer.ready:
            continue

        entropy_coef = mappo_brain.entropy_coefficient(episode_index)
        update_started = time.perf_counter()
        update_metrics = mappo_brain.update(buffer.consume(), entropy_coef)
        ppo_seconds = time.perf_counter() - update_started
        ppo_seconds_total += ppo_seconds
        diagnostics["update"].append(update_metrics["update"])
        diagnostics["update_episode"].append(episode_index + 1)
        _append(diagnostics["update_metrics"], update_metrics, update_keys)
        diagnostics["runtime"]["rollout_seconds_per_update"].append(
            rollout_seconds_since_update
        )
        diagnostics["runtime"]["ppo_update_seconds"].append(ppo_seconds)
        diagnostics["runtime"]["elapsed_seconds"].append(
            time.perf_counter() - run_started
        )
        diagnostics["runtime"]["simulation_steps_per_update"].append(
            simulation_steps_since_update
        )
        diagnostics["runtime"]["transitions_per_update"].append(
            int(update_metrics["transition_count"])
        )

        completed_episode = episode_index + 1
        completed_episode_count = completed_episode
        if (
            args.common_checkpoint_episode > 0
            and completed_episode == args.common_checkpoint_episode
        ):
            _atomic_torch_save(
                mappo_brain.actor.state_dict(),
                run_dir / f"actor_ep{completed_episode}.pt",
            )
        should_validate = (
            args.validation_every_updates > 0
            and completed_episode >= args.validation_start_episode
            and mappo_brain.update_count % args.validation_every_updates == 0
        )
        should_checkpoint = (
            args.checkpoint_every_updates > 0
            and mappo_brain.update_count % args.checkpoint_every_updates == 0
        )
        if should_validate:
            validation = evaluate_fixed_validation_seeds(
                env_config, mappo_brain, args.validation_seeds,
                conditions=mixed_conditions,
            )
            validation.update({
                "episode": completed_episode,
                "update": mappo_brain.update_count,
                "seeds": list(args.validation_seeds),
            })
            is_best = (
                validation["mean_episode_return_per_agent"]
                > best_validation_return
            )
            validation["is_best"] = is_best
            joint_eligible = (
                joint_fixed_baseline is not None
                and validation["failure_count"]
                <= joint_fixed_baseline["failure_count"]
                and validation["mean_episode_return_per_agent"]
                > joint_fixed_baseline["mean_episode_return_per_agent"]
                and validation["mean_arrival_time_with_timeouts"]
                < joint_fixed_baseline["mean_arrival_time_with_timeouts"]
            )
            joint_is_best = joint_eligible and (
                validation["mean_episode_return_per_agent"]
                > best_joint_validation_return
                or (
                    validation["mean_episode_return_per_agent"]
                    == best_joint_validation_return
                    and validation["mean_arrival_time_with_timeouts"]
                    < best_joint_validation_arrival
                )
            )
            validation["joint_eligible"] = joint_eligible
            validation["joint_is_best"] = joint_is_best
            if joint_fixed_baseline is not None:
                validation["return_margin_over_fixed"] = (
                    validation["mean_episode_return_per_agent"]
                    - joint_fixed_baseline["mean_episode_return_per_agent"]
                )
                validation["arrival_margin_over_fixed"] = (
                    joint_fixed_baseline["mean_arrival_time_with_timeouts"]
                    - validation["mean_arrival_time_with_timeouts"]
                )
            else:
                validation["return_margin_over_fixed"] = None
                validation["arrival_margin_over_fixed"] = None
            robust_joint_eligible = (
                joint_fixed_baseline is not None
                and validation["failure_count"]
                <= joint_fixed_baseline["failure_count"]
                and validation["return_margin_over_fixed"]
                >= args.robust_joint_min_return_margin
                and validation["arrival_margin_over_fixed"]
                >= args.robust_joint_min_arrival_margin
            )
            robust_joint_is_best = robust_joint_eligible and (
                validation["mean_episode_return_per_agent"]
                > best_robust_joint_validation_return
                or (
                    validation["mean_episode_return_per_agent"]
                    == best_robust_joint_validation_return
                    and validation["mean_arrival_time_with_timeouts"]
                    < best_robust_joint_validation_arrival
                )
            )
            validation["robust_joint_eligible"] = robust_joint_eligible
            validation["robust_joint_is_best"] = robust_joint_is_best

            # Safety-first checkpoint selection across every validation episode:
            # minimize worst-condition/seed failures, then mean failures, then
            # maximize return, and finally minimize all-agent arrival time.
            safe_key = (
                int(validation["max_failure_count"]),
                float(validation["failure_count"]),
                -float(validation["mean_episode_return_per_agent"]),
                float(validation["mean_arrival_time_with_timeouts"]),
            )
            safe_state = diagnostics["safe_model_selection"]
            previous_safe_key = safe_state.get("best_key")
            safe_is_best = (
                previous_safe_key is None
                or safe_key < tuple(previous_safe_key)
            )
            validation["safe_key"] = list(safe_key)
            validation["safe_is_best"] = bool(safe_is_best)
            safe_state.setdefault("checks", []).append({
                "episode": completed_episode,
                "safe_key": list(safe_key),
                "is_best": bool(safe_is_best),
            })
            if safe_is_best:
                safe_state["best_key"] = list(safe_key)
                safe_state["best_episode"] = completed_episode
                _atomic_torch_save(
                    mappo_brain.actor.state_dict(),
                    run_dir / "best_safe_actor.pt",
                )

            if args.early_stopping:
                best_eligible_return = early_stop_state[
                    "best_eligible_return"
                ]
                best_eligible_arrival = early_stop_state[
                    "best_eligible_arrival"
                ]
                material_improvement = False
                if robust_joint_eligible:
                    material_improvement = (
                        best_eligible_return is None
                        or best_eligible_arrival is None
                        or validation["mean_episode_return_per_agent"]
                        > best_eligible_return
                        + args.early_stop_return_improvement
                        or validation["mean_arrival_time_with_timeouts"]
                        < best_eligible_arrival
                        - args.early_stop_arrival_improvement
                    )
                    early_stop_state["best_eligible_return"] = max(
                        validation["mean_episode_return_per_agent"],
                        best_eligible_return
                        if best_eligible_return is not None
                        else -float("inf"),
                    )
                    early_stop_state["best_eligible_arrival"] = min(
                        validation["mean_arrival_time_with_timeouts"],
                        best_eligible_arrival
                        if best_eligible_arrival is not None
                        else float("inf"),
                    )
                robust_checkpoint_available = (
                    robust_joint_is_best
                    or (run_dir / "best_robust_joint_actor.pt").exists()
                )
                minimum_reached = (
                    completed_episode >= args.early_stop_min_episode
                )
                if not minimum_reached or material_improvement:
                    early_stop_state["plateau_checks"] = 0
                elif robust_checkpoint_available:
                    early_stop_state["plateau_checks"] += 1
                else:
                    early_stop_state["plateau_checks"] = 0

                stability = _training_stability(
                    diagnostics,
                    args.early_stop_window,
                    args.early_stop_return_tolerance,
                    args.early_stop_arrival_tolerance,
                )
                validation_safe = validation["failure_count"] == 0.0
                stop_training = bool(
                    minimum_reached
                    and stability["training_stable"]
                    and validation_safe
                    and robust_joint_eligible
                    and robust_checkpoint_available
                    and early_stop_state["plateau_checks"]
                    >= args.early_stop_patience
                )
                early_stop_check = {
                    "episode": completed_episode,
                    **stability,
                    "validation_safe": bool(validation_safe),
                    "robust_joint_eligible": bool(robust_joint_eligible),
                    "robust_checkpoint_available": bool(
                        robust_checkpoint_available
                    ),
                    "material_improvement": bool(material_improvement),
                    "plateau_checks": int(
                        early_stop_state["plateau_checks"]
                    ),
                    "minimum_reached": bool(minimum_reached),
                    "should_stop": bool(stop_training),
                }
                early_stop_state["checks"].append(early_stop_check)
                validation["early_stop_check"] = early_stop_check
                if stop_training:
                    stop_reason = (
                        "training stable over two "
                        f"{args.early_stop_window}-episode windows; "
                        f"zero failures on {len(args.validation_seeds)} "
                        "validation seeds; robust checkpoint available; "
                        f"no material improvement for "
                        f"{early_stop_state['plateau_checks']} checks"
                    )
                    early_stop_state["stopped"] = True
                    early_stop_state["stop_episode"] = completed_episode
                    early_stop_state["reason"] = stop_reason
            diagnostics["validation"].append(validation)
            if is_best:
                best_validation_return = validation[
                    "mean_episode_return_per_agent"
                ]
                _atomic_torch_save(
                    mappo_brain.actor.state_dict(), run_dir / "best_actor.pt"
                )
                # actor.pt remains the default consumed by existing evaluation
                # scripts, but now represents the best validation checkpoint.
                _atomic_torch_save(
                    mappo_brain.actor.state_dict(), run_dir / "actor.pt"
                )
            if joint_is_best:
                best_joint_validation_return = validation[
                    "mean_episode_return_per_agent"
                ]
                best_joint_validation_arrival = validation[
                    "mean_arrival_time_with_timeouts"
                ]
                _atomic_torch_save(
                    mappo_brain.actor.state_dict(), run_dir / "best_joint_actor.pt"
                )
            if robust_joint_is_best:
                best_robust_joint_validation_return = validation[
                    "mean_episode_return_per_agent"
                ]
                best_robust_joint_validation_arrival = validation[
                    "mean_arrival_time_with_timeouts"
                ]
                _atomic_torch_save(
                    mappo_brain.actor.state_dict(),
                    run_dir / "best_robust_joint_actor.pt",
                )
            _atomic_torch_save(
                _checkpoint_payload(
                    completed_episode, env, mappo_brain, diagnostics,
                    best_validation_return, best_joint_validation_return,
                    best_joint_validation_arrival,
                    best_robust_joint_validation_return,
                    best_robust_joint_validation_arrival,
                ),
                run_dir / "latest_checkpoint.pt",
            )
            robust_status = (
                "best" if robust_joint_is_best
                else ("eligible" if robust_joint_eligible else "no")
            )
            print(
                f"validation episode={completed_episode:05d} "
                f"return={validation['mean_episode_return_per_agent']:.3f} "
                f"arrival_time={validation['mean_arrival_time_with_timeouts']:.1f}s "
                f"far={validation['sampled_far_fraction']:.3f} "
                f"failures={validation['failure_count']:.2f} "
                f"max_failure={validation['max_failure_count']} "
                f"safe_best={'yes' if safe_is_best else 'no'} "
                f"best={'yes' if is_best else 'no'} "
                f"joint={'best' if joint_is_best else ('eligible' if joint_eligible else 'no')} "
                f"robust_joint={robust_status}",
                flush=True,
            )
            if args.early_stopping:
                check = early_stop_state["checks"][-1]
                print(
                    f"early_stop stable={check['training_stable']} "
                    f"delta_return={check['return_change']:.3f} "
                    f"delta_arrival={check['arrival_change']:.3f}s "
                    f"safe_validation={check['validation_safe']} "
                    f"robust={check['robust_checkpoint_available']} "
                    f"plateau={check['plateau_checks']}/"
                    f"{args.early_stop_patience} "
                    f"stop={check['should_stop']}",
                    flush=True,
                )
        elif should_checkpoint:
            _atomic_torch_save(
                _checkpoint_payload(
                    completed_episode, env, mappo_brain, diagnostics,
                    best_validation_return, best_joint_validation_return,
                    best_joint_validation_arrival,
                    best_robust_joint_validation_return,
                    best_robust_joint_validation_arrival,
                ),
                run_dir / "latest_checkpoint.pt",
            )

        if mappo_brain.update_count % args.log_every_updates == 0:
            mean_reward = float(np.mean([item["mean_agent_reward"] for item in recent_metrics]))
            mean_arrival = float(np.mean([item["mean_arrival_time_arrived_only"] for item in recent_metrics]))
            mean_arrived = int(round(np.mean([item["arrival_count"] for item in recent_metrics])))
            mean_total = int(round(np.mean([
                item["total_count"] for item in recent_metrics
            ])))
            demand_labels = sorted({
                int(item["total_count"]) for item in recent_metrics
            })
            blockage_episodes = sum(item["blockage_edge"] is not None for item in recent_metrics)
            mean_failures = float(np.mean([item["failure_count"] for item in recent_metrics]))
            mean_rejections = float(np.mean([
                item["capacity_rejection_count"] for item in recent_metrics
            ]))
            mean_timeouts = float(np.mean([item["timeout_count"] for item in recent_metrics]))
            capacity_labels = sorted({
                int(item["near_shelter_capacity"]) for item in recent_metrics
            })
            print(
                f"episode={episode_index + 1:05d}/{args.episodes} update={mappo_brain.update_count:05d} "
                f"reward={mean_reward:.3f} arrival_time={mean_arrival:.1f}s "
                f"arrived={mean_arrived}/{mean_total} "
                f"sampled_far={update_metrics['sampled_far_fraction']:.3f} "
                f"prob_far={update_metrics['mean_prob_far']:.3f} "
                f"policy_entropy={update_metrics['policy_entropy']:.4f} "
                f"entropy_coef={update_metrics['entropy_coefficient']:.4f} "
                f"actor_loss={update_metrics['actor_loss']:.4f} "
                f"critic_loss={update_metrics['critic_loss']:.4f} "
                f"approx_kl={update_metrics['approx_kl']:.6f} "
                f"clip_fraction={update_metrics['clip_fraction']:.4f}",
                f"demand={demand_labels}",
                f"near_capacity={capacity_labels}",
                f"failures={mean_failures:.1f}",
                f"capacity_rejections={mean_rejections:.1f}",
                f"timeouts={mean_timeouts:.1f}",
                f"blockage_episodes={blockage_episodes}/{ppo_config.rollout_episodes}",
                flush=True,
            )
            if args.profile_runtime:
                print(
                    f"runtime rollout_{ppo_config.rollout_episodes}ep="
                    f"{rollout_seconds_since_update:.3f}s "
                    f"ppo_update={ppo_seconds:.3f}s "
                    f"elapsed={time.perf_counter() - run_started:.1f}s "
                    f"simulation_steps={simulation_steps_since_update}",
                    flush=True,
                )
        recent_metrics.clear()
        rollout_seconds_since_update = 0.0
        simulation_steps_since_update = 0
        if stop_training:
            print(
                f"Early stopping at episode {completed_episode}: "
                f"{stop_reason}",
                flush=True,
            )
            break

    if rollout_executor is not None:
        rollout_executor.shutdown(wait=True)

    diagnostics["unused_complete_episodes"] = buffer.episode_count
    total_wall_seconds = time.perf_counter() - run_started
    episodes_this_run = completed_episode_count - start_episode
    runtime_summary = {
        "profile_runtime": bool(args.profile_runtime),
        "parallel_rollout_workers": int(args.parallel_rollout_workers),
        "torch_threads": int(torch.get_num_threads()),
        "total_wall_seconds": total_wall_seconds,
        "rollout_seconds": rollout_seconds_total,
        "ppo_update_seconds": ppo_seconds_total,
        "other_seconds": max(
            total_wall_seconds - rollout_seconds_total - ppo_seconds_total, 0.0
        ),
        "episodes": int(completed_episode_count),
        "requested_max_episodes": int(args.episodes),
        "episodes_this_run": int(episodes_this_run),
        "ppo_updates": int(mappo_brain.update_count),
        "simulation_steps": int(simulation_steps_total),
        "transitions_collected": int(transitions_total),
        "seconds_per_episode": (
            total_wall_seconds / episodes_this_run
            if episodes_this_run > 0 else 0.0
        ),
        "environment_steps_per_second": (
            simulation_steps_total / rollout_seconds_total
            if rollout_seconds_total > 0.0 else 0.0
        ),
        "transitions_per_wall_second": (
            transitions_total / total_wall_seconds
            if total_wall_seconds > 0.0 else 0.0
        ),
    }
    diagnostics["runtime_summary"] = runtime_summary
    if args.validation_every_updates == 0 or not diagnostics["validation"]:
        _atomic_torch_save(mappo_brain.actor.state_dict(), run_dir / "actor.pt")
        _atomic_torch_save(mappo_brain.actor.state_dict(), run_dir / "best_actor.pt")
    if buffer.episode_count == 0:
        _atomic_torch_save(
            _checkpoint_payload(
                completed_episode_count, env, mappo_brain, diagnostics,
                best_validation_return, best_joint_validation_return,
                best_joint_validation_arrival,
                best_robust_joint_validation_return,
                best_robust_joint_validation_arrival,
            ),
            run_dir / "latest_checkpoint.pt",
        )
    torch.save(mappo_brain.critic.state_dict(), run_dir / "critic.pt")
    with (run_dir / "diagnostics.pkl").open("wb") as handle:
        pickle.dump(diagnostics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (run_dir / "environment_config.json").write_text(
        json.dumps(asdict(env_config), indent=2), encoding="utf-8"
    )
    (run_dir / "ppo_config.json").write_text(
        json.dumps(asdict(ppo_config), indent=2), encoding="utf-8"
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps({
            "seed": args.seed,
            "entropy_mode": args.entropy_mode,
            "entropy_coefficient": args.entropy_coefficient,
            "entropy_start": args.entropy_start,
            "entropy_end": args.entropy_end,
            "entropy_anneal_episodes": args.entropy_anneal_episodes,
            "team_return_gamma": args.team_return_gamma,
            "capacity_mode": args.capacity_mode,
            "capacity_observation": selected_capacity_observation,
            "num_agent_candidates": args.num_agent_candidates,
            "near_capacity_ratios": args.near_capacity_ratios,
            "far_capacity_ratio": args.far_capacity_ratio,
            "demand_observation_scale": args.demand_observation_scale,
            "mixed_demand_conditions": [
                list(condition) for condition in mixed_conditions
            ],
            "profile_runtime": args.profile_runtime,
            "parallel_rollout_workers": args.parallel_rollout_workers,
            "torch_threads": int(torch.get_num_threads()),
            "validation_every_updates": args.validation_every_updates,
            "checkpoint_every_updates": args.checkpoint_every_updates,
            "validation_start_episode": args.validation_start_episode,
            "validation_seeds": args.validation_seeds,
            "joint_fixed_far_probabilities": args.joint_fixed_far_probabilities,
            "skip_joint_fixed_baseline": args.skip_joint_fixed_baseline,
            "joint_fixed_baseline": joint_fixed_baseline,
            "robust_joint_min_return_margin": args.robust_joint_min_return_margin,
            "robust_joint_min_arrival_margin": args.robust_joint_min_arrival_margin,
            "requested_max_episodes": args.episodes,
            "completed_episodes": completed_episode_count,
            "common_checkpoint_episode": args.common_checkpoint_episode,
            "early_stopping": early_stop_state,
            "best_validation_return": (
                best_validation_return
                if np.isfinite(best_validation_return) else None
            ),
            "best_joint_validation_return": (
                best_joint_validation_return
                if np.isfinite(best_joint_validation_return) else None
            ),
            "best_joint_validation_arrival": (
                best_joint_validation_arrival
                if np.isfinite(best_joint_validation_arrival) else None
            ),
            "has_best_joint_actor": (run_dir / "best_joint_actor.pt").exists(),
            "best_robust_joint_validation_return": (
                best_robust_joint_validation_return
                if np.isfinite(best_robust_joint_validation_return) else None
            ),
            "best_robust_joint_validation_arrival": (
                best_robust_joint_validation_arrival
                if np.isfinite(best_robust_joint_validation_arrival) else None
            ),
            "has_best_robust_joint_actor": (
                run_dir / "best_robust_joint_actor.pt"
            ).exists(),
            "safe_model_selection": diagnostics.get("safe_model_selection"),
            "has_best_safe_actor": (run_dir / "best_safe_actor.pt").exists(),
            "resumed_from": str(args.resume.resolve()) if args.resume else None,
        }, indent=2),
        encoding="utf-8",
    )
    (run_dir / "runtime_summary.json").write_text(
        json.dumps(runtime_summary, indent=2), encoding="utf-8"
    )
    if buffer.episode_count:
        print(
            f"Skipped final {buffer.episode_count} episode(s): a PPO update requires exactly "
            f"{ppo_config.rollout_episodes} complete episodes.", flush=True,
        )
    print(
        f"Runtime summary: total={total_wall_seconds:.1f}s "
        f"rollout={rollout_seconds_total:.1f}s "
        f"ppo={ppo_seconds_total:.1f}s "
        f"seconds_per_episode={runtime_summary['seconds_per_episode']:.4f}",
        flush=True,
    )
    print(f"Saved new run to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
