"""Evacuation environment (legacy filename, stable two-edge dynamics)."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from common import (
    CONGESTION_PENALTY,
    DEPARTURE_FRACTION,
    DT,
    FAR_ROAD_LENGTH,
    FREE_FLOW_DENSITY,
    MAX_BASE_SPEED,
    MAX_DENSITY,
    MAX_STEPS_EP,
    MIN_BASE_SPEED,
    MIN_SPEED_FACTOR,
    NEAR_ROAD_LENGTH,
    N_TRAIN_AGENTS,
    ROAD_WIDTH,
    TIME_PENALTY,
)

# Status codes keep the explicit naming convention used by the original files.
STATUS_WAITING = 0
STATUS_NEAR_EDGE = 1
STATUS_FAR_EDGE = 2
STATUS_ARRIVED = 3
STATUS_FAILED = 4

@dataclass(frozen=True)
class EvacuationConfig:
    num_agents: int = N_TRAIN_AGENTS
    max_steps: int = MAX_STEPS_EP
    dt: float = DT
    departure_window_fraction: float = DEPARTURE_FRACTION
    near_length: float = NEAR_ROAD_LENGTH
    far_length: float = FAR_ROAD_LENGTH
    road_width: float = ROAD_WIDTH
    free_density: float = FREE_FLOW_DENSITY
    max_density: float = MAX_DENSITY
    minimum_speed_factor: float = MIN_SPEED_FACTOR
    min_base_speed: float = MIN_BASE_SPEED
    max_base_speed: float = MAX_BASE_SPEED
    congestion_penalty: float = CONGESTION_PENALTY
    time_penalty: float = TIME_PENALTY
    closure_edge: Optional[int] = None
    closure_start_step: int = 0
    closure_end_step: int = 0
    closure_density_floor: float = 1.0
    closure_capacity_fraction: float = 0.0
    closure_speed_factor: float = 0.0
    random_blockage_probability: float = 0.0
    random_blockage_min_duration: int = 30
    random_blockage_max_duration: int = 180
    near_shelter_capacity: Optional[int] = None
    far_shelter_capacity: Optional[int] = None
    near_shelter_capacity_candidates: tuple[int, ...] = ()
    failure_penalty: float = 0.0
    include_capacity_observation: bool = True
    include_shelter_capacity_observation: bool = False
    shelter_capacity_mode: str = "hard_mask"
    observation_schema: str = "legacy"


class EvacuationEnv:
    """All agents make one irreversible shelter choice, then walk to completion.

    State codes are: 0 waiting at center, 1 on near, 2 on far,
    3 arrived, and 4 failed. Shelter capacity is reserved irreversibly when an
    action is committed; an arrival does not release its reservation.
    """

    def __init__(self, config: Optional[EvacuationConfig] = None, seed: int = 0):
        self.config = config or EvacuationConfig()
        self._validate_config()
        self.rng = np.random.default_rng(seed)
        self.lengths = np.asarray(
            [self.config.near_length, self.config.far_length], dtype=np.float32
        )
        self.areas = self.lengths * self.config.road_width
        self.reset()

    def _capacity_options(self, configured: Optional[int], candidates: Sequence[int]) -> tuple[int, ...]:
        if candidates:
            return tuple(int(value) for value in candidates)
        return (self.config.num_agents if configured is None else int(configured),)

    def _validate_config(self) -> None:
        c = self.config
        if c.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if c.shelter_capacity_mode not in ("hard_mask", "reject"):
            raise ValueError("shelter_capacity_mode must be 'hard_mask' or 'reject'")
        if c.observation_schema not in (
            "legacy", "capacity_ratio", "capacity_absolute_and_ratio"
        ):
            raise ValueError(
                "observation_schema must be 'legacy', 'capacity_ratio', or "
                "'capacity_absolute_and_ratio'"
            )
        if c.observation_schema in ("capacity_ratio", "capacity_absolute_and_ratio") and (
            c.near_shelter_capacity is None
            and not c.near_shelter_capacity_candidates
            and c.far_shelter_capacity is None
        ):
            raise ValueError("capacity_ratio observations require shelter capacities")
        if c.include_shelter_capacity_observation and not c.include_capacity_observation:
            raise ValueError(
                "Shelter-capacity observations require availability observations"
            )
        near_options = self._capacity_options(
            c.near_shelter_capacity, c.near_shelter_capacity_candidates
        )
        far_options = self._capacity_options(c.far_shelter_capacity, ())
        if any(value < 0 for value in (*near_options, *far_options)):
            raise ValueError("Shelter capacities must be non-negative")
        if any(near + far_options[0] < c.num_agents for near in near_options):
            raise ValueError(
                "Every shelter-capacity combination must accommodate all agents"
            )

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.config
        departure_steps = max(1, int(np.ceil(c.max_steps * c.departure_window_fraction)))
        self.episode_closure_edge = c.closure_edge
        self.episode_closure_start_step = c.closure_start_step
        self.episode_closure_end_step = c.closure_end_step
        self.episode_closure_capacity_fraction = c.closure_capacity_fraction
        self.episode_closure_speed_factor = c.closure_speed_factor
        if c.random_blockage_probability > 0.0 and self.rng.random() < c.random_blockage_probability:
            self.episode_closure_edge = int(self.rng.integers(0, 2))
            self.episode_closure_start_step = int(self.rng.integers(0, departure_steps))
            duration = int(self.rng.integers(
                c.random_blockage_min_duration,
                c.random_blockage_max_duration + 1,
            ))
            self.episode_closure_end_step = min(
                c.max_steps, self.episode_closure_start_step + duration
            )
            self.episode_closure_capacity_fraction = 0.0
            self.episode_closure_speed_factor = 0.0
        self.start_steps = self.rng.integers(0, departure_steps, c.num_agents)
        near_options = self._capacity_options(
            c.near_shelter_capacity, c.near_shelter_capacity_candidates
        )
        self.episode_shelter_capacities = np.asarray(
            [int(self.rng.choice(near_options)), self._capacity_options(c.far_shelter_capacity, ())[0]],
            dtype=np.int64,
        )
        self.shelter_reserved = np.zeros(2, dtype=np.int64)
        self.base_speeds = self.rng.uniform(
            c.min_base_speed, c.max_base_speed, c.num_agents
        ).astype(np.float32)
        self.status = np.zeros(c.num_agents, dtype=np.int8)
        self.capacity_rejected = np.zeros(c.num_agents, dtype=bool)
        self.action = np.full(c.num_agents, -1, dtype=np.int8)
        self.distance = np.zeros(c.num_agents, dtype=np.float32)
        self.raw_returns = np.zeros(c.num_agents, dtype=np.float32)
        self.discounted_returns = np.zeros(c.num_agents, dtype=np.float32)
        self.discount_multiplier = np.ones(c.num_agents, dtype=np.float32)
        # Mean reward across all agents at each environment step. This preserves
        # the reported system objective while allowing a global reward-to-go to
        # be assigned to every shelter-choice decision.
        self.team_step_rewards = np.zeros(c.max_steps, dtype=np.float32)
        self.arrival_step = np.full(c.num_agents, np.nan, dtype=np.float32)
        self.travel_time = np.full(c.num_agents, np.nan, dtype=np.float32)
        self.previous_density = np.zeros(2, dtype=np.float32)
        self.current_step = 0

    def closure_active(self) -> bool:
        return (
            self.episode_closure_edge in (0, 1)
            and self.episode_closure_start_step
            <= self.current_step
            < self.episode_closure_end_step
        )

    def edge_availability(self) -> np.ndarray:
        """Return edge availability, including a temporary road closure."""
        capacities = np.ones(2, dtype=np.float32)
        if self.closure_active():
            capacities[int(self.episode_closure_edge)] = (
                self.episode_closure_capacity_fraction
            )
        return capacities

    def capacity_fractions(self) -> np.ndarray:
        """Backward-compatible name for edge availability used by 5-D actors."""
        return self.edge_availability()

    def remaining_shelter_capacity(self) -> np.ndarray:
        return self.episode_shelter_capacities - self.shelter_reserved

    def remaining_shelter_capacity_fractions(self) -> np.ndarray:
        """Remaining spaces normalized by total episode demand, not own capacity."""
        return self.remaining_shelter_capacity().astype(np.float32) / self.config.num_agents

    def remaining_shelter_capacity_ratios(self) -> np.ndarray:
        """Fraction of currently waiting demand each remaining capacity can hold."""
        waiting = max(int(np.count_nonzero(self.status == 0)), 1)
        return np.clip(
            self.remaining_shelter_capacity().astype(np.float32) / waiting,
            0.0,
            1.0,
        )

    def action_mask(self) -> np.ndarray:
        available = self.edge_availability() > 0.0
        if self.config.shelter_capacity_mode == "reject":
            return available
        return available & (self.remaining_shelter_capacity() > 0)

    def observed_density(self) -> np.ndarray:
        """Density exposed to the policy, including an optional closure signal."""
        density = self.previous_density.copy()
        if self.closure_active():
            edge = int(self.episode_closure_edge)
            density[edge] = max(density[edge], self.config.closure_density_floor)
        return density

    def normalized_density(self) -> np.ndarray:
        return np.clip(self.observed_density() / self.config.max_density, 0.0, 1.0)

    def observations_for(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build observations for a specified set of currently departing agents."""
        ids = np.asarray(ids, dtype=np.int64)
        density = self.normalized_density()
        waiting_fraction = np.count_nonzero(self.status == 0) / self.config.num_agents
        if self.config.observation_schema in (
            "capacity_ratio", "capacity_absolute_and_ratio"
        ):
            ratios = self.remaining_shelter_capacity_ratios()
            include_absolute = (
                self.config.observation_schema == "capacity_absolute_and_ratio"
            )
            obs = np.empty((len(ids), 7 if include_absolute else 5), dtype=np.float32)
            obs[:, :2] = density
            obs[:, 2] = waiting_fraction
            if include_absolute:
                obs[:, 3:5] = self.remaining_shelter_capacity_fractions()
                obs[:, 5:7] = ratios
            else:
                obs[:, 3:5] = ratios
            state = np.empty((len(ids), 9 if include_absolute else 7), dtype=np.float32)
            state[:, :2] = density
            state[:, 2] = waiting_fraction
            state[:, 3] = np.count_nonzero(
                (self.status == 1) | (self.status == 2)
            ) / self.config.num_agents
            state[:, 4] = np.count_nonzero(self.status == 4) / self.config.num_agents
            if include_absolute:
                state[:, 5:7] = self.remaining_shelter_capacity_fractions()
                state[:, 7:9] = ratios
            else:
                state[:, 5:7] = ratios
            return obs, state
        observation_dim = 3
        if self.config.include_capacity_observation:
            observation_dim += 2
        if self.config.include_shelter_capacity_observation:
            observation_dim += 2
        obs = np.empty((len(ids), observation_dim), dtype=np.float32)
        obs[:, :2] = density
        # Include agents departing on this step: they are still waiting when the
        # simultaneous decision batch is formed.
        obs[:, 2] = waiting_fraction
        state = np.empty((len(ids), observation_dim), dtype=np.float32)
        state[:, :2] = density
        state[:, 2] = np.count_nonzero(self.status < 3) / self.config.num_agents
        if self.config.include_capacity_observation:
            availability = self.edge_availability()
            obs[:, 3:5] = availability
            state[:, 3:5] = availability
        if self.config.include_shelter_capacity_observation:
            remaining = self.remaining_shelter_capacity_fractions()
            obs[:, 5:7] = remaining
            state[:, 5:7] = remaining
        return obs, state

    def decision_batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return departing IDs, decentralized observations, and critic states."""
        ids = np.flatnonzero(
            (self.status == 0) & (self.start_steps == self.current_step)
        )
        obs, state = self.observations_for(ids)
        return ids, obs, state

    def commit(self, ids: np.ndarray, actions: np.ndarray) -> None:
        if len(ids) != len(actions):
            raise ValueError("One action is required for every departing agent")
        if np.any((actions < 0) | (actions > 1)):
            raise ValueError("Actions must be 0 (near) or 1 (far)")
        if np.any(self.status[ids] != 0):
            raise RuntimeError("Only waiting agents may reserve a shelter")
        availability = self.edge_availability()
        if np.any(availability[actions] <= 0.0):
            raise RuntimeError("An action selected a currently unavailable edge")
        self.action[ids] = actions.astype(np.int8)
        if self.config.shelter_capacity_mode == "reject":
            remaining = self.remaining_shelter_capacity()
            for edge in (0, 1):
                selected = ids[actions == edge]
                slots = int(remaining[edge])
                if len(selected) > slots:
                    selected = self.rng.permutation(selected)
                accepted = selected[:slots]
                rejected = selected[slots:]
                self.shelter_reserved[edge] += len(accepted)
                self.status[accepted] = edge + 1
                self.status[rejected] = 4
                self.capacity_rejected[rejected] = True
            return
        reservations = np.bincount(actions, minlength=2)
        if np.any(reservations > self.remaining_shelter_capacity()):
            raise RuntimeError("Shelter capacity would be exceeded by this decision batch")
        self.shelter_reserved += reservations
        self.status[ids] = actions.astype(np.int8) + 1

    def advance(self, gamma: float) -> None:
        """Apply reward and movement for one second, using lagged density for speed."""
        c = self.config
        active = (self.status == 1) | (self.status == 2)
        if np.any(active):
            edge = self.status[active] - 1

            # Reward uses current occupancy, including agents that just departed.
            counts = np.bincount(edge, minlength=2)
            current_density = counts / self.areas
            if self.closure_active():
                closed_edge = int(self.episode_closure_edge)
                current_density[closed_edge] = max(
                    current_density[closed_edge], c.closure_density_floor
                )
            ratio = current_density[edge] / c.max_density
            excess = np.maximum(0.0, ratio - c.free_density / c.max_density)
            excess /= 1.0 - c.free_density / c.max_density
            rewards = -c.congestion_penalty * np.minimum(excess, 1.0) - c.time_penalty
            active_ids = np.flatnonzero(active)
            self.team_step_rewards[self.current_step] = rewards.sum() / c.num_agents
            self.raw_returns[active_ids] += rewards.astype(np.float32)
            self.discounted_returns[active_ids] += (
                self.discount_multiplier[active_ids] * rewards
            ).astype(np.float32)
            self.discount_multiplier[active_ids] *= gamma

            # Movement deliberately uses the density from the previous step.
            lagged_ratio = self.previous_density[edge] / c.max_density
            blend = np.clip(
                (lagged_ratio - c.free_density / c.max_density)
                / (1.0 - c.free_density / c.max_density),
                0.0,
                1.0,
            )
            speed_factor = 1.0 - (1.0 - c.minimum_speed_factor) * blend
            if self.closure_active():
                speed_factor[edge == int(self.episode_closure_edge)] = (
                    self.episode_closure_speed_factor
                )
            self.distance[active_ids] += self.base_speeds[active_ids] * speed_factor * c.dt
            arrived = self.distance[active_ids] >= self.lengths[edge]
            arrived_ids = active_ids[arrived]
            if len(arrived_ids):
                self.status[arrived_ids] = 3
                self.arrival_step[arrived_ids] = (self.current_step + 1) * c.dt
                self.travel_time[arrived_ids] = (
                    self.current_step + 1 - self.start_steps[arrived_ids]
                ) * c.dt

            # Density for the next step is measured after arrivals leave the roads.
            near_count = np.count_nonzero(self.status == 1)
            far_count = np.count_nonzero(self.status == 2)
            self.previous_density = np.asarray(
                [near_count / self.areas[0], far_count / self.areas[1]],
                dtype=np.float32,
            )
        else:
            self.previous_density.fill(0.0)
        self.current_step += 1

    def team_discounted_reward_to_go(self, gamma: float) -> np.ndarray:
        """Return system mean reward-to-go starting at every environment step."""
        result = np.zeros(self.config.max_steps, dtype=np.float32)
        running = 0.0
        for step in range(self.current_step - 1, -1, -1):
            running = float(self.team_step_rewards[step]) + gamma * running
            result[step] = running
        return result

    def finished(self) -> bool:
        return self.current_step >= self.config.max_steps or np.all(self.status >= 3)

    def finalize(self) -> Dict[str, object]:
        timed_out = self.status < 3
        self.status[timed_out] = 4
        failed = self.status == 4
        if np.any(failed) and self.config.failure_penalty != 0.0:
            self.raw_returns[failed] += self.config.failure_penalty
        arrived = self.status == 3
        timed_arrivals = np.where(arrived, self.travel_time, self.config.max_steps)
        chosen = self.action >= 0
        return {
            "mean_agent_reward": float(self.raw_returns.mean()),
            "total_reward": float(self.raw_returns.sum()),
            "arrival_count": int(arrived.sum()),
            "failure_count": int((self.status == 4).sum()),
            "capacity_rejection_count": int(self.capacity_rejected.sum()),
            "timeout_count": int(timed_out.sum()),
            "mean_arrival_time_arrived_only": (
                float(np.nanmean(self.travel_time[arrived])) if np.any(arrived) else float("nan")
            ),
            "mean_arrival_time_all_agents": float(timed_arrivals.mean()),
            "far_fraction": float(np.mean(self.action[chosen] == 1)) if np.any(chosen) else 0.0,
            "per_agent_arrival_times": self.travel_time.copy(),
            "per_agent_rewards": self.raw_returns.copy(),
            "near_shelter_capacity": int(self.episode_shelter_capacities[0]),
            "far_shelter_capacity": int(self.episode_shelter_capacities[1]),
            "near_reserved_count": int(self.shelter_reserved[0]),
            "far_reserved_count": int(self.shelter_reserved[1]),
            "team_failure_penalty": float(
                self.config.failure_penalty * np.mean(failed)
            ),
            "per_agent_failed": failed.copy(),
            "blockage_edge": self.episode_closure_edge,
            "blockage_start_step": self.episode_closure_start_step,
            "blockage_end_step": self.episode_closure_end_step,
        }
