"""Training entry point using the original project layout and naming.

All environment, return-target, advantage, and PPO computations match
the validated Codex implementation; only organization/names differ.
"""

"""From-scratch shared-actor, centralized-critic PPO implementation."""

from dataclasses import dataclass
from typing import Dict

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
    TOTAL_EPISODES,
)
from torch.distributions import Categorical

import argparse
import json
import pickle
import random
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
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_EP)
    parser.add_argument("--road-width", type=float, default=5.0)
    parser.add_argument("--entropy-mode", choices=("constant", "annealed"), default="constant")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--log-every-updates", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=512)
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


def collect_episode(env: EvacuationEnv, mappo_brain: MAPPOAgent) -> tuple[dict, dict]:
    observations, states, actions, action_masks, log_probs, probabilities, agent_ids = [], [], [], [], [], [], []
    env.reset()
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

    ids = np.concatenate(agent_ids)
    order = np.argsort(ids)
    obs = np.concatenate(observations)[order]
    state = np.concatenate(states)[order]
    action = np.concatenate(actions)[order]
    masks = np.concatenate(action_masks)[order]
    old_log_prob = np.concatenate(log_probs)[order]
    probs = np.concatenate(probabilities)[order]
    decision_steps = env.start_steps[ids][order]
    team_reward_to_go = env.team_discounted_reward_to_go(mappo_brain.config.gamma)
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
    metrics["total_count"] = env.config.num_agents
    return batch, metrics


def _append(mapping: dict[str, list], record: dict, keys: tuple[str, ...]) -> None:
    for key in keys:
        mapping[key].append(record[key])


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.log_every_updates <= 0:
        raise ValueError("--log-every-updates must be positive")
    if not 0.0 <= args.blockage_probability <= 1.0:
        raise ValueError("--blockage-probability must be in [0, 1]")
    if not 1 <= args.blockage_min_duration <= args.blockage_max_duration:
        raise ValueError("Blockage durations must satisfy 1 <= min <= max")
    if args.near_capacities is not None and any(value < 0 for value in args.near_capacities):
        raise ValueError("--near-capacities values must be non-negative")
    if args.far_capacity is not None and args.far_capacity < 0:
        raise ValueError("--far-capacity must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    capacity_enabled = args.near_capacities is not None or args.far_capacity is not None
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
    env_config = EvacuationConfig(
        num_agents=args.num_agents,
        max_steps=args.max_steps,
        road_width=args.road_width,
        random_blockage_probability=args.blockage_probability,
        random_blockage_min_duration=args.blockage_min_duration,
        random_blockage_max_duration=args.blockage_max_duration,
        far_shelter_capacity=args.far_capacity,
        near_shelter_capacity_candidates=tuple(args.near_capacities or ()),
        failure_penalty=failure_penalty,
        include_capacity_observation=True,
        include_shelter_capacity_observation=capacity_enabled,
        shelter_capacity_mode=args.capacity_mode.replace("-", "_"),
        observation_schema={
            "absolute": "legacy",
            "demand-ratio": "capacity_ratio",
            "absolute-and-demand-ratio": "capacity_absolute_and_ratio",
        }[args.capacity_observation],
    )
    if args.capacity_observation == "demand-ratio":
        actor_input_dim, critic_input_dim = 5, 7
    elif args.capacity_observation == "absolute-and-demand-ratio":
        actor_input_dim, critic_input_dim = 7, 9
    else:
        actor_input_dim = critic_input_dim = 7 if capacity_enabled else 5
    ppo_config = PPOConfig(
        entropy_mode=args.entropy_mode,
        minibatch_size=args.minibatch_size,
        observation_dim=actor_input_dim,
        critic_state_dim=critic_input_dim,
    )
    env = EvacuationEnv(env_config, seed=args.seed)
    mappo_brain = MAPPOAgent(ppo_config, device)
    buffer = RolloutBuffer(ppo_config.rollout_episodes)
    run_dir = make_new_run_directory(args.output_dir, args.entropy_mode, args.seed)

    episode_keys = (
        "mean_episode_return_per_agent", "arrived_only_mean_arrival_time",
        "mean_arrival_time_with_timeouts", "arrived_count", "total_count", "failure_count",
        "sampled_far_fraction", "mean_prob_near", "mean_prob_far", "policy_entropy",
        "mean_training_return_target",
        "near_shelter_capacity", "far_shelter_capacity",
        "near_reserved_count", "far_reserved_count", "team_failure_penalty",
        "capacity_rejection_count", "timeout_count",
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
        "episode": list(range(1, args.episodes + 1)),
        "episode_metrics": {key: [] for key in episode_keys},
        "update": [],
        "update_episode": [],
        "update_metrics": {key: [] for key in update_keys},
    }
    recent_metrics: list[dict] = []

    for episode_index in range(args.episodes):
        batch, metrics = collect_episode(env, mappo_brain)
        buffer.add(batch)
        recent_metrics.append(metrics)
        _append(diagnostics["episode_metrics"], metrics, episode_keys)
        if not buffer.ready:
            continue

        entropy_coef = mappo_brain.entropy_coefficient(episode_index)
        update_metrics = mappo_brain.update(buffer.consume(), entropy_coef)
        diagnostics["update"].append(update_metrics["update"])
        diagnostics["update_episode"].append(episode_index + 1)
        _append(diagnostics["update_metrics"], update_metrics, update_keys)

        if mappo_brain.update_count % args.log_every_updates == 0:
            mean_reward = float(np.mean([item["mean_agent_reward"] for item in recent_metrics]))
            mean_arrival = float(np.mean([item["mean_arrival_time_arrived_only"] for item in recent_metrics]))
            mean_arrived = int(round(np.mean([item["arrival_count"] for item in recent_metrics])))
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
                f"arrived={mean_arrived}/{args.num_agents} "
                f"sampled_far={update_metrics['sampled_far_fraction']:.3f} "
                f"prob_far={update_metrics['mean_prob_far']:.3f} "
                f"policy_entropy={update_metrics['policy_entropy']:.4f} "
                f"entropy_coef={update_metrics['entropy_coefficient']:.4f} "
                f"actor_loss={update_metrics['actor_loss']:.4f} "
                f"critic_loss={update_metrics['critic_loss']:.4f} "
                f"approx_kl={update_metrics['approx_kl']:.6f} "
                f"clip_fraction={update_metrics['clip_fraction']:.4f}",
                f"near_capacity={capacity_labels}",
                f"failures={mean_failures:.1f}",
                f"capacity_rejections={mean_rejections:.1f}",
                f"timeouts={mean_timeouts:.1f}",
                f"blockage_episodes={blockage_episodes}/{ppo_config.rollout_episodes}",
                flush=True,
            )
        recent_metrics.clear()

    diagnostics["unused_complete_episodes"] = buffer.episode_count
    torch.save(mappo_brain.actor.state_dict(), run_dir / "actor.pt")
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
            "capacity_mode": args.capacity_mode,
            "capacity_observation": args.capacity_observation,
        }, indent=2),
        encoding="utf-8",
    )
    if buffer.episode_count:
        print(
            f"Skipped final {buffer.episode_count} episode(s): a PPO update requires exactly "
            f"{ppo_config.rollout_episodes} complete episodes.", flush=True,
        )
    print(f"Saved new run to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
