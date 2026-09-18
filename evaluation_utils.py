"""Shared helpers for evaluating saved evacuation actors without training."""

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from evac_env import EvacuationConfig, EvacuationEnv
from training import Actor, masked_probabilities


def resolve_actor(path: Path) -> Path:
    if path.is_file():
        if path.name != "actor.pt":
            raise ValueError(f"Expected actor.pt, got: {path}")
        return path
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    candidates = list(path.rglob("actor.pt"))
    if not candidates:
        raise FileNotFoundError(f"No actor.pt found under: {path}")
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def load_actor_and_config(path: Path) -> tuple[Actor, EvacuationConfig, Path]:
    actor_path = resolve_actor(path)
    config_path = actor_path.with_name("environment_config.json")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(EvacuationConfig)}
    config = EvacuationConfig(**{key: value for key, value in raw_config.items() if key in allowed})
    state_dict = torch.load(actor_path, map_location="cpu", weights_only=True)
    input_dim = int(state_dict["network.0.weight"].shape[1])
    actor = Actor(input_dim)
    actor.load_state_dict(state_dict)
    actor.eval()
    if config.observation_schema == "legacy":
        config = replace(
            config,
            include_capacity_observation=input_dim >= 5,
            include_shelter_capacity_observation=input_dim >= 7,
        )
    elif config.observation_schema == "capacity_ratio" and input_dim != 5:
        raise ValueError(
            f"capacity_ratio environment requires a 5-D Actor, got {input_dim}"
        )
    elif config.observation_schema == "capacity_absolute_and_ratio" and input_dim != 7:
        raise ValueError(
            "capacity_absolute_and_ratio environment requires a 7-D Actor, "
            f"got {input_dim}"
        )
    return actor, config, actor_path


def run_evaluation_episode(
    config: EvacuationConfig,
    seed: int,
    actor: Optional[Actor] = None,
    fixed_far_probability: Optional[float] = None,
    collect_trace: bool = False,
) -> dict:
    """Run one policy episode; exactly one of actor/fixed probability is required."""
    if (actor is None) == (fixed_far_probability is None):
        raise ValueError("Specify exactly one of actor or fixed_far_probability")
    if fixed_far_probability is not None and not 0.0 <= fixed_far_probability <= 1.0:
        raise ValueError("fixed_far_probability must be in [0, 1]")
    if actor is not None and config.observation_schema == "legacy":
        config = replace(
            config,
            include_capacity_observation=actor.input_dim >= 5,
            include_shelter_capacity_observation=actor.input_dim >= 7,
        )
    env = EvacuationEnv(config, seed=seed)
    env.reset(seed=seed)
    action_rng = np.random.default_rng(seed + 1_000_003)
    raw_far_probability_total = 0.0
    effective_far_probability_total = 0.0
    probability_count = 0
    trace = {
        "physical_density_near": [],
        "observed_density_near": [],
        "far_decisions": np.zeros(config.max_steps, dtype=np.int64),
        "far_probability_sum": np.zeros(config.max_steps, dtype=np.float64),
        "decision_count": np.zeros(config.max_steps, dtype=np.int64),
        "decision_density_near": [],
        "decision_density_far": [],
        "decision_raw_density_near": [],
        "decision_raw_density_far": [],
        "decision_prob_far_raw": [],
        "decision_prob_far_effective": [],
        "decision_actions": [],
        "decision_remaining_waiting_fraction": [],
        "decision_remaining_near_capacity_fraction": [],
        "decision_remaining_far_capacity_fraction": [],
        "decision_action_was_forced": [],
    }
    while not env.finished():
        step = env.current_step
        ids, observations, _ = env.decision_batch()
        trace["observed_density_near"].append(float(env.observed_density()[0]))
        if len(ids):
            remaining = env.remaining_shelter_capacity()
            boundary_batch = np.any((remaining > 0) & (remaining < len(ids)))
            groups = (
                [np.asarray([agent_id]) for agent_id in env.rng.permutation(ids)]
                if boundary_batch else [ids]
            )
            for group_ids in groups:
                if boundary_batch:
                    observations, _ = env.observations_for(group_ids)
                mask = env.action_mask()
                remaining_capacity_before = env.remaining_shelter_capacity_fractions()
                masks = np.repeat(mask[None, :], len(group_ids), axis=0)
                if actor is not None:
                    with torch.no_grad():
                        observation_tensor = torch.from_numpy(observations)
                        raw_distribution = actor(observation_tensor)
                        effective_distribution = masked_probabilities(
                            raw_distribution, torch.from_numpy(masks)
                        )
                        raw_far_probabilities = raw_distribution[:, 1].numpy()
                        probabilities = effective_distribution[:, 1].numpy()
                else:
                    base = np.asarray(
                        [1.0 - float(fixed_far_probability), float(fixed_far_probability)]
                    )
                    raw_far_probabilities = np.full(len(group_ids), base[1])
                    effective = base * mask
                    if effective.sum() <= 0.0:
                        # Deterministic nominal policies (p=0 or p=1) can assign
                        # zero probability to the only shelter left. Feasibility
                        # takes precedence once their preferred shelter is full.
                        effective = mask.astype(np.float64)
                    if effective.sum() <= 0.0:
                        raise RuntimeError("No available shelter for fixed-policy evaluation")
                    probabilities = np.full(len(group_ids), effective[1] / effective.sum())
                actions = (action_rng.random(len(group_ids)) < probabilities).astype(np.int8)
                env.commit(group_ids, actions)
                raw_far_probability_total += float(raw_far_probabilities.sum())
                effective_far_probability_total += float(probabilities.sum())
                probability_count += len(group_ids)
                trace["far_decisions"][step] += int(actions.sum())
                trace["far_probability_sum"][step] += float(probabilities.sum())
                trace["decision_count"][step] += len(actions)
                if collect_trace:
                    trace["decision_density_near"].extend(observations[:, 0].tolist())
                    trace["decision_density_far"].extend(observations[:, 1].tolist())
                    trace["decision_raw_density_near"].extend(
                        np.full(len(group_ids), env.previous_density[0]).tolist()
                    )
                    trace["decision_raw_density_far"].extend(
                        np.full(len(group_ids), env.previous_density[1]).tolist()
                    )
                    trace["decision_prob_far_raw"].extend(raw_far_probabilities.tolist())
                    trace["decision_prob_far_effective"].extend(probabilities.tolist())
                    trace["decision_actions"].extend(actions.tolist())
                    trace["decision_remaining_waiting_fraction"].extend(
                        observations[:, 2].tolist()
                    )
                    trace["decision_remaining_near_capacity_fraction"].extend(
                        np.full(len(group_ids), remaining_capacity_before[0]).tolist()
                    )
                    trace["decision_remaining_far_capacity_fraction"].extend(
                        np.full(len(group_ids), remaining_capacity_before[1]).tolist()
                    )
                    trace["decision_action_was_forced"].extend(
                        np.full(len(group_ids), np.count_nonzero(mask) == 1).tolist()
                    )
        env.advance(gamma=0.99)
        trace["physical_density_near"].append(float(env.previous_density[0]))
    metrics = env.finalize()
    result = {
        "seed": seed,
        "mean_episode_return_per_agent": metrics["mean_agent_reward"],
        "arrived_only_mean_arrival_time": metrics["mean_arrival_time_arrived_only"],
        "mean_arrival_time_with_timeouts": metrics["mean_arrival_time_all_agents"],
        "arrived_count": metrics["arrival_count"],
        "total_count": config.num_agents,
        "sampled_far_fraction": metrics["far_fraction"],
        "mean_actor_probability_far": (
            raw_far_probability_total / probability_count if probability_count else float("nan")
        ),
        "mean_effective_probability_far": (
            effective_far_probability_total / probability_count
            if probability_count else float("nan")
        ),
        "failure_count": metrics["failure_count"],
        "capacity_rejection_count": metrics["capacity_rejection_count"],
        "timeout_count": metrics["timeout_count"],
        "near_shelter_capacity": metrics["near_shelter_capacity"],
        "far_shelter_capacity": metrics["far_shelter_capacity"],
        "near_reserved_count": metrics["near_reserved_count"],
        "far_reserved_count": metrics["far_reserved_count"],
    }
    if collect_trace:
        result["trace"] = trace
    return result


def aggregate_numeric(rows: list[dict], keys: tuple[str, ...]) -> dict:
    return {key: float(np.nanmean([row[key] for row in rows])) for key in keys}
