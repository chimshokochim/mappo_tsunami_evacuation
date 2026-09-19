"""Train capacity-free MAPPO on the three-shelter star environment."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from three_shelter_env import SHELTER_NAMES, ThreeShelterConfig, ThreeShelterEnv


def require_finite(name: str, value) -> None:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if not np.all(np.isfinite(array)):
        bad = np.argwhere(~np.isfinite(array))
        first = tuple(bad[0]) if bad.size else ()
        raise FloatingPointError(f"NaN/Inf in {name}; first bad index={first}")


class Actor(nn.Module):
    def __init__(self, input_dim: int = 4, action_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim), nn.Softmax(dim=-1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class Critic(nn.Module):
    def __init__(self, input_dim: int = 7):
        super().__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


@dataclass(frozen=True)
class PPOConfig:
    observation_dim: int = 4
    critic_state_dim: int = 7
    action_dim: int = 3
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 512
    rollout_episodes: int = 4
    max_grad_norm: float = 0.5
    entropy_mode: str = "constant"
    entropy_coefficient: float = 0.05
    entropy_start: float = 0.05
    entropy_end: float = 0.01
    entropy_anneal_episodes: int = 9000
    advantage_epsilon: float = 1e-5
    advantage_reward: str = "team_mean_reward_to_go"


def generalized_advantage_estimate(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE kept explicit; each choice here is a terminal macro-transition."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    for name, value in (("reward", rewards), ("value", values),
                        ("next value", next_values), ("done", dones)):
        require_finite(name, value)
    delta = rewards + gamma * next_values * (1.0 - dones) - values
    # Transitions are independent terminal decisions, so no sample may
    # bootstrap into another agent or episode. Lambda therefore has no
    # numerical effect, but the standard equation remains explicit.
    advantages = delta.copy()
    returns = advantages + values
    require_finite("advantage", advantages)
    require_finite("return", returns)
    return advantages, returns


class RolloutBuffer:
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
            raise RuntimeError("RolloutBuffer is full; consume before adding")
        self._episodes.append(episode)

    def consume(self) -> dict[str, np.ndarray]:
        if not self.ready:
            raise RuntimeError(
                f"Need {self.episodes_per_update} complete episodes, have {self.episode_count}"
            )
        keys = self._episodes[0].keys()
        result = {
            key: np.concatenate([episode[key] for episode in self._episodes])
            for key in keys
        }
        self._episodes.clear()
        return result


class MAPPOAgent:
    def __init__(self, config: PPOConfig, device: torch.device):
        self.config = config
        self.device = device
        self.actor = Actor(config.observation_dim, config.action_dim).to(device)
        self.critic = Critic(config.critic_state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )
        self.update_count = 0

    @torch.no_grad()
    def act(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        probabilities = self.actor(obs)
        require_finite("action probabilities", probabilities)
        distribution = Categorical(probabilities)
        actions = distribution.sample()
        return (
            actions.cpu().numpy(), distribution.log_prob(actions).cpu().numpy(),
            probabilities.cpu().numpy(),
        )

    def entropy_coefficient(self, episode: int) -> float:
        c = self.config
        if c.entropy_mode == "constant":
            return c.entropy_coefficient
        fraction = min(max(episode / float(c.entropy_anneal_episodes), 0.0), 1.0)
        return c.entropy_start + fraction * (c.entropy_end - c.entropy_start)

    def update(self, batch: dict[str, np.ndarray], entropy_coef: float) -> dict[str, float]:
        c = self.config
        obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(
            batch["log_probs"], dtype=torch.float32, device=self.device
        ).detach()
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device).detach()
        for name, value in (("observation", obs), ("state", states),
                            ("old log probability", old_log_probs), ("return", returns)):
            require_finite(name, value)

        with torch.no_grad():
            old_values = self.critic(states)
            advantages = (returns - old_values).detach()
            advantages = (
                advantages - advantages.mean()
            ) / (advantages.std(unbiased=False) + c.advantage_epsilon)
        require_finite("normalized advantage", advantages)

        size = len(actions)
        sums = {
            "actor_loss": 0.0, "critic_loss": 0.0, "policy_entropy": 0.0,
            "mean_prob_near": 0.0, "mean_prob_middle": 0.0,
            "mean_prob_far": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0,
        }
        sample_count = 0
        for _ in range(c.ppo_epochs):
            # Exactly one permutation per PPO epoch. Every transition appears
            # once in that epoch, split sequentially into minibatches.
            permutation = torch.randperm(size, device=self.device)
            for start in range(0, size, c.minibatch_size):
                ids = permutation[start:start + c.minibatch_size]
                probabilities = self.actor(obs[ids])
                distribution = Categorical(probabilities)
                new_log_probs = distribution.log_prob(actions[ids])
                entropy = distribution.entropy()
                ratio = torch.exp(new_log_probs - old_log_probs[ids])
                surr1 = ratio * advantages[ids]
                surr2 = torch.clamp(
                    ratio, 1.0 - c.clip_epsilon, 1.0 + c.clip_epsilon
                ) * advantages[ids]
                actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy.mean()
                require_finite("probability ratio", ratio)
                require_finite("actor loss", actor_loss)

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm)
                self.actor_optimizer.step()

                predicted_values = self.critic(states[ids])
                critic_loss = 0.5 * ((predicted_values - returns[ids]) ** 2).mean()
                require_finite("critic prediction", predicted_values)
                require_finite("critic loss", critic_loss)
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs[ids] - new_log_probs).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > c.clip_epsilon
                    ).float().mean()
                values = {
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "policy_entropy": entropy.mean(),
                    "mean_prob_near": probabilities[:, 0].mean(),
                    "mean_prob_middle": probabilities[:, 1].mean(),
                    "mean_prob_far": probabilities[:, 2].mean(),
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                }
                n = len(ids)
                for key, value in values.items():
                    require_finite(key, value)
                    sums[key] += float(value.item()) * n
                sample_count += n

        self.update_count += 1
        result = {key: value / sample_count for key, value in sums.items()}
        result.update({
            "sampled_near_fraction": float((actions == 0).float().mean().item()),
            "sampled_middle_fraction": float((actions == 1).float().mean().item()),
            "sampled_far_fraction": float((actions == 2).float().mean().item()),
            "entropy_coefficient": float(entropy_coef),
            "transition_count": size,
            "update": self.update_count,
        })
        for name, value in result.items():
            require_finite(name, value)
        return result


def collect_episode(
    env: ThreeShelterEnv, agent: MAPPOAgent
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    observations, states, actions, log_probs, probabilities, agent_ids = [], [], [], [], [], []
    env.reset()
    while not env.finished():
        ids, obs, global_states = env.decision_batch()
        if len(ids):
            sampled_actions, sampled_log_probs, action_probabilities = agent.act(obs)
            env.commit(ids, sampled_actions)
            observations.append(obs)
            states.append(global_states)
            actions.append(sampled_actions)
            log_probs.append(sampled_log_probs)
            probabilities.append(action_probabilities)
            agent_ids.append(ids)
        env.advance(agent.config.gamma)
    metrics = env.finalize()

    ids = np.concatenate(agent_ids)
    order = np.argsort(ids)
    obs = np.concatenate(observations)[order]
    state = np.concatenate(states)[order]
    action = np.concatenate(actions)[order]
    old_log_prob = np.concatenate(log_probs)[order]
    probs = np.concatenate(probabilities)[order]
    decision_steps = env.start_steps[ids][order]
    team_reward_to_go = env.team_discounted_reward_to_go(agent.config.gamma)
    discounted_return = team_reward_to_go[decision_steps]

    with torch.no_grad():
        values = agent.critic(
            torch.as_tensor(state, dtype=torch.float32, device=agent.device)
        ).cpu().numpy()
    _, return_targets = generalized_advantage_estimate(
        discounted_return, values, np.zeros_like(values), np.ones_like(values),
        agent.config.gamma, agent.config.gae_lambda,
    )
    batch = {
        "observations": obs,
        "states": state,
        "actions": action,
        "log_probs": old_log_prob,
        "returns": return_targets,
    }
    metrics.update({
        "mean_prob_near": float(probs[:, 0].mean()),
        "mean_prob_middle": float(probs[:, 1].mean()),
        "mean_prob_far": float(probs[:, 2].mean()),
        "policy_entropy": float(
            (-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)).mean()
        ),
        "mean_training_return_target": float(np.mean(return_targets)),
        "mean_episode_return_per_agent": metrics["mean_agent_reward"],
        "arrived_only_mean_arrival_time": metrics["mean_arrival_time_arrived_only"],
        "mean_arrival_time_with_timeouts": metrics["mean_arrival_time_all_agents"],
        "arrived_count": metrics["arrival_count"],
        "total_count": env.config.num_agents,
    })
    return batch, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes", type=int,
        help="total target episodes (new runs default to 12000; resume keeps the saved target)",
    )
    parser.add_argument("--num-agents", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--road-lengths", type=float, nargs=3, default=(150.0, 225.0, 300.0))
    parser.add_argument("--road-width", type=float, default=5.0)
    parser.add_argument("--entropy-mode", choices=("constant", "annealed"), default="annealed")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_three_shelter"))
    parser.add_argument("--log-every-updates", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument(
        "--checkpoint-every-updates", type=int, default=25,
        help="atomically replace one rolling checkpoint every N PPO updates; 0 disables",
    )
    parser.add_argument(
        "--resume", type=Path,
        help="run directory or checkpoint_latest.pt to resume in place",
    )
    parser.add_argument(
        "--stop-after-updates", type=int,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(name)


def make_run_directory(parent: Path, mode: str, seed: int) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    stem = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{mode}_seed{seed}"
    run_dir = parent / stem
    run_dir.mkdir()
    return run_dir


def resolve_checkpoint(path: Path) -> Path:
    checkpoint = path / "checkpoint_latest.pt" if path.is_dir() else path
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def save_checkpoint(
    checkpoint_path: Path,
    *,
    next_episode: int,
    target_episodes: int,
    env: ThreeShelterEnv,
    agent: MAPPOAgent,
    diagnostics: dict,
    seed: int,
) -> None:
    """Atomically replace the single rolling checkpoint.

    This is called only immediately after a PPO update, when the rollout
    buffer is empty. A partial four-episode rollout is therefore never mixed
    with a resumed run.
    """
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    payload = {
        "checkpoint_version": 1,
        "next_episode": int(next_episode),
        "target_episodes": int(target_episodes),
        "seed": int(seed),
        "environment_config": asdict(env.config),
        "ppo_config": asdict(agent.config),
        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
        "update_count": int(agent.update_count),
        "diagnostics": diagnostics,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "environment_random_state": env.rng.bit_generator.state,
    }
    torch.save(payload, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def restore_checkpoint(
    checkpoint: dict,
    env: ThreeShelterEnv,
    agent: MAPPOAgent,
) -> None:
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("Unsupported checkpoint version")
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic.load_state_dict(checkpoint["critic_state_dict"])
    agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    agent.update_count = int(checkpoint["update_count"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    if torch.cuda.is_available() and checkpoint["torch_cuda_random_state_all"] is not None:
        torch.cuda.set_rng_state_all([
            state.cpu() for state in checkpoint["torch_cuda_random_state_all"]
        ])
    env.rng.bit_generator.state = checkpoint["environment_random_state"]


def append_metrics(target: dict[str, list], source: dict, keys: tuple[str, ...]) -> None:
    for key in keys:
        target[key].append(source[key])


def main() -> None:
    args = parse_args()
    if args.episodes is not None and args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.num_agents <= 0 or args.max_steps <= 0:
        raise ValueError("num-agents and max-steps must be positive")
    if args.log_every_updates <= 0 or args.minibatch_size <= 0:
        raise ValueError("log-every-updates and minibatch-size must be positive")
    if args.checkpoint_every_updates < 0:
        raise ValueError("checkpoint-every-updates must be non-negative")
    if args.stop_after_updates is not None and args.stop_after_updates <= 0:
        raise ValueError("stop-after-updates must be positive")

    device = choose_device(args.device)
    checkpoint_data = None
    resumed_from = None
    if args.resume is not None:
        checkpoint_path = resolve_checkpoint(args.resume)
        checkpoint_data = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        seed = int(checkpoint_data["seed"])
        total_episodes = int(
            checkpoint_data["target_episodes"] if args.episodes is None else args.episodes
        )
        env_config = ThreeShelterConfig(**checkpoint_data["environment_config"])
        ppo_config = PPOConfig(**checkpoint_data["ppo_config"])
        run_dir = checkpoint_path.parent
        resumed_from = str(checkpoint_path.resolve())
    else:
        seed = args.seed
        total_episodes = 12000 if args.episodes is None else args.episodes
        env_config = ThreeShelterConfig(
            num_agents=args.num_agents,
            max_steps=args.max_steps,
            road_lengths=tuple(args.road_lengths),
            road_width=args.road_width,
        )
        ppo_config = PPOConfig(
            entropy_mode=args.entropy_mode,
            minibatch_size=args.minibatch_size,
        )
        run_dir = make_run_directory(args.output_dir, args.entropy_mode, seed)
        checkpoint_path = run_dir / "checkpoint_latest.pt"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = ThreeShelterEnv(env_config, seed=seed)
    agent = MAPPOAgent(ppo_config, device)
    buffer = RolloutBuffer(ppo_config.rollout_episodes)

    episode_keys = (
        "mean_episode_return_per_agent", "arrived_only_mean_arrival_time",
        "mean_arrival_time_with_timeouts", "arrived_count", "total_count",
        "failure_count", "timeout_count", "sampled_near_fraction",
        "sampled_middle_fraction", "sampled_far_fraction", "mean_prob_near",
        "mean_prob_middle", "mean_prob_far", "policy_entropy",
        "mean_training_return_target",
    )
    update_keys = (
        "actor_loss", "critic_loss", "policy_entropy", "entropy_coefficient",
        "mean_prob_near", "mean_prob_middle", "mean_prob_far",
        "sampled_near_fraction", "sampled_middle_fraction", "sampled_far_fraction",
        "approx_kl", "clip_fraction", "transition_count",
    )
    if checkpoint_data is not None:
        restore_checkpoint(checkpoint_data, env, agent)
        diagnostics = checkpoint_data["diagnostics"]
        diagnostics["device"] = str(device)
        start_episode = int(checkpoint_data["next_episode"])
        if start_episode >= total_episodes:
            raise ValueError(
                f"Checkpoint already reached episode {start_episode}; "
                f"requested total is {total_episodes}"
            )
        print(
            f"Resuming {checkpoint_path.resolve()} at episode {start_episode + 1}/"
            f"{total_episodes}, update={agent.update_count}",
            flush=True,
        )
    else:
        start_episode = 0
        diagnostics = {
            "environment_config": asdict(env_config),
            "ppo_config": asdict(ppo_config),
            "seed": seed,
            "entropy_mode": ppo_config.entropy_mode,
            "device": str(device),
            "shelters": list(SHELTER_NAMES),
            "episode": [],
            "episode_metrics": {key: [] for key in episode_keys},
            "update": [], "update_episode": [],
            "update_metrics": {key: [] for key in update_keys},
        }
    recent_metrics: list[dict] = []
    updates_this_invocation = 0
    for episode_index in range(start_episode, total_episodes):
        batch, metrics = collect_episode(env, agent)
        buffer.add(batch)
        recent_metrics.append(metrics)
        diagnostics["episode"].append(episode_index + 1)
        append_metrics(diagnostics["episode_metrics"], metrics, episode_keys)
        if not buffer.ready:
            continue
        entropy_coef = agent.entropy_coefficient(episode_index)
        update_metrics = agent.update(buffer.consume(), entropy_coef)
        diagnostics["update"].append(update_metrics["update"])
        diagnostics["update_episode"].append(episode_index + 1)
        append_metrics(diagnostics["update_metrics"], update_metrics, update_keys)
        updates_this_invocation += 1

        if agent.update_count % args.log_every_updates == 0:
            reward = float(np.mean([row["mean_agent_reward"] for row in recent_metrics]))
            arrival = float(np.mean([
                row["mean_arrival_time_arrived_only"] for row in recent_metrics
            ]))
            arrived = int(round(np.mean([row["arrival_count"] for row in recent_metrics])))
            print(
                f"episode={episode_index + 1:05d}/{total_episodes} "
                f"update={agent.update_count:05d} reward={reward:.3f} "
                f"arrival_time={arrival:.1f}s arrived={arrived}/{env_config.num_agents} "
                f"sampled=(near={update_metrics['sampled_near_fraction']:.3f},"
                f"mid={update_metrics['sampled_middle_fraction']:.3f},"
                f"far={update_metrics['sampled_far_fraction']:.3f}) "
                f"prob=(near={update_metrics['mean_prob_near']:.3f},"
                f"mid={update_metrics['mean_prob_middle']:.3f},"
                f"far={update_metrics['mean_prob_far']:.3f}) "
                f"policy_entropy={update_metrics['policy_entropy']:.4f} "
                f"entropy_coef={update_metrics['entropy_coefficient']:.4f} "
                f"actor_loss={update_metrics['actor_loss']:.4f} "
                f"critic_loss={update_metrics['critic_loss']:.4f} "
                f"approx_kl={update_metrics['approx_kl']:.6f} "
                f"clip_fraction={update_metrics['clip_fraction']:.4f}",
                flush=True,
            )
        recent_metrics.clear()

        periodic_checkpoint = (
            args.checkpoint_every_updates > 0
            and agent.update_count % args.checkpoint_every_updates == 0
        )
        requested_pause = (
            args.stop_after_updates is not None
            and updates_this_invocation >= args.stop_after_updates
        )
        if periodic_checkpoint or requested_pause:
            save_checkpoint(
                checkpoint_path,
                next_episode=episode_index + 1,
                target_episodes=total_episodes,
                env=env,
                agent=agent,
                diagnostics=diagnostics,
                seed=seed,
            )
            print(
                f"Saved rolling checkpoint: {checkpoint_path.resolve()} "
                f"(next episode {episode_index + 2})",
                flush=True,
            )
        if requested_pause:
            print("Stopped at a safe post-update checkpoint by request.", flush=True)
            return

    diagnostics["unused_complete_episodes"] = buffer.episode_count
    torch.save(agent.actor.state_dict(), run_dir / "actor.pt")
    torch.save(agent.critic.state_dict(), run_dir / "critic.pt")
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
            "topology": "three_shelter_star",
            "capacity_limits": False,
            "road_blockage": False,
            "seed": seed,
            "entropy_mode": ppo_config.entropy_mode,
            "resumed_from": resumed_from,
            "checkpoint_every_updates": args.checkpoint_every_updates,
        }, indent=2), encoding="utf-8",
    )
    if buffer.episode_count:
        print(
            f"Skipped final {buffer.episode_count} episode(s): an update requires "
            f"{ppo_config.rollout_episodes} complete episodes.", flush=True,
        )
    checkpoint_path.unlink(missing_ok=True)
    checkpoint_path.with_name(checkpoint_path.name + ".tmp").unlink(missing_ok=True)
    print(f"Saved new run to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
