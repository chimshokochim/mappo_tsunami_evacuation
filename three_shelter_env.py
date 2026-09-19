"""Capacity-free three-shelter star environment.

Every agent makes one irreversible choice at departure:
0 = near, 1 = middle, 2 = far.  It then travels on that road without making
another decision.  This deliberately preserves the terminal macro-transition
used by the validated two-shelter experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


STATUS_WAITING = 0
STATUS_ARRIVED = 4
STATUS_FAILED = 5
SHELTER_NAMES = ("near", "middle", "far")


@dataclass(frozen=True)
class ThreeShelterConfig:
    num_agents: int = 3000
    max_steps: int = 900
    dt: float = 1.0
    departure_window_fraction: float = 0.25
    road_lengths: tuple[float, float, float] = (150.0, 225.0, 300.0)
    road_width: float = 5.0
    free_density: float = 0.1
    max_density: float = 1.0
    minimum_speed_factor: float = 0.3
    min_base_speed: float = 1.0
    max_base_speed: float = 1.5
    congestion_penalty: float = 0.50
    time_penalty: float = 0.01


class ThreeShelterEnv:
    action_count = 3
    observation_dim = 4
    critic_state_dim = 7

    def __init__(self, config: Optional[ThreeShelterConfig] = None, seed: int = 0):
        self.config = config or ThreeShelterConfig()
        self._validate_config()
        self.rng = np.random.default_rng(seed)
        self.lengths = np.asarray(self.config.road_lengths, dtype=np.float32)
        self.areas = self.lengths * self.config.road_width
        self.reset()

    def _validate_config(self) -> None:
        c = self.config
        if c.num_agents <= 0 or c.max_steps <= 0:
            raise ValueError("num_agents and max_steps must be positive")
        if len(c.road_lengths) != self.action_count:
            raise ValueError("road_lengths must contain near, middle, and far lengths")
        if any(length <= 0 for length in c.road_lengths) or c.road_width <= 0:
            raise ValueError("Road lengths and width must be positive")
        if not 0.0 < c.departure_window_fraction <= 1.0:
            raise ValueError("departure_window_fraction must be in (0, 1]")
        if not 0.0 <= c.free_density < c.max_density:
            raise ValueError("Require 0 <= free_density < max_density")
        if not 0.0 < c.minimum_speed_factor <= 1.0:
            raise ValueError("minimum_speed_factor must be in (0, 1]")
        if not 0.0 < c.min_base_speed <= c.max_base_speed:
            raise ValueError("Invalid base-speed range")

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.config
        departure_steps = max(1, int(np.ceil(c.max_steps * c.departure_window_fraction)))
        self.start_steps = self.rng.integers(0, departure_steps, c.num_agents)
        self.base_speeds = self.rng.uniform(
            c.min_base_speed, c.max_base_speed, c.num_agents
        ).astype(np.float32)
        self.status = np.full(c.num_agents, STATUS_WAITING, dtype=np.int8)
        self.action = np.full(c.num_agents, -1, dtype=np.int8)
        self.distance = np.zeros(c.num_agents, dtype=np.float32)
        self.raw_returns = np.zeros(c.num_agents, dtype=np.float32)
        self.discounted_returns = np.zeros(c.num_agents, dtype=np.float32)
        self.discount_multiplier = np.ones(c.num_agents, dtype=np.float32)
        self.team_step_rewards = np.zeros(c.max_steps, dtype=np.float32)
        self.arrival_step = np.full(c.num_agents, np.nan, dtype=np.float32)
        self.travel_time = np.full(c.num_agents, np.nan, dtype=np.float32)
        self.previous_density = np.zeros(self.action_count, dtype=np.float32)
        self.current_step = 0

    def normalized_density(self) -> np.ndarray:
        return np.clip(
            self.previous_density / self.config.max_density, 0.0, 1.0
        )

    def action_mask(self) -> np.ndarray:
        return np.ones(self.action_count, dtype=bool)

    def observations_for(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ids = np.asarray(ids, dtype=np.int64)
        density = self.normalized_density()
        waiting_fraction = np.count_nonzero(self.status == STATUS_WAITING) / self.config.num_agents
        obs = np.empty((len(ids), self.observation_dim), dtype=np.float32)
        obs[:, :3] = density
        obs[:, 3] = waiting_fraction

        state = np.empty((len(ids), self.critic_state_dim), dtype=np.float32)
        state[:, :3] = density
        for edge in range(self.action_count):
            state[:, 3 + edge] = np.count_nonzero(self.status == edge + 1) / self.config.num_agents
        state[:, 6] = waiting_fraction
        return obs, state

    def decision_batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids = np.flatnonzero(
            (self.status == STATUS_WAITING) & (self.start_steps == self.current_step)
        )
        obs, state = self.observations_for(ids)
        return ids, obs, state

    def commit(self, ids: np.ndarray, actions: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        actions = np.asarray(actions, dtype=np.int64)
        if len(ids) != len(actions):
            raise ValueError("One action is required for every departing agent")
        if np.any((actions < 0) | (actions >= self.action_count)):
            raise ValueError("Actions must be 0 (near), 1 (middle), or 2 (far)")
        if np.any(self.status[ids] != STATUS_WAITING):
            raise RuntimeError("Only waiting agents may choose a shelter")
        self.action[ids] = actions.astype(np.int8)
        self.status[ids] = actions.astype(np.int8) + 1

    def advance(self, gamma: float) -> None:
        c = self.config
        active = (self.status >= 1) & (self.status <= self.action_count)
        if np.any(active):
            active_ids = np.flatnonzero(active)
            edge = self.status[active_ids] - 1
            counts = np.bincount(edge, minlength=self.action_count)
            current_density = counts / self.areas

            ratio = current_density[edge] / c.max_density
            excess = np.maximum(0.0, ratio - c.free_density / c.max_density)
            excess /= 1.0 - c.free_density / c.max_density
            rewards = -c.congestion_penalty * np.minimum(excess, 1.0) - c.time_penalty
            self.team_step_rewards[self.current_step] = rewards.sum() / c.num_agents
            self.raw_returns[active_ids] += rewards.astype(np.float32)
            self.discounted_returns[active_ids] += (
                self.discount_multiplier[active_ids] * rewards
            ).astype(np.float32)
            self.discount_multiplier[active_ids] *= gamma

            # As in the two-shelter model, movement uses the previous step density.
            lagged_ratio = self.previous_density[edge] / c.max_density
            blend = np.clip(
                (lagged_ratio - c.free_density / c.max_density)
                / (1.0 - c.free_density / c.max_density),
                0.0,
                1.0,
            )
            speed_factor = 1.0 - (1.0 - c.minimum_speed_factor) * blend
            self.distance[active_ids] += (
                self.base_speeds[active_ids] * speed_factor * c.dt
            )
            arrived = self.distance[active_ids] >= self.lengths[edge]
            arrived_ids = active_ids[arrived]
            if len(arrived_ids):
                self.status[arrived_ids] = STATUS_ARRIVED
                self.arrival_step[arrived_ids] = (self.current_step + 1) * c.dt
                self.travel_time[arrived_ids] = (
                    self.current_step + 1 - self.start_steps[arrived_ids]
                ) * c.dt

            next_counts = np.asarray(
                [np.count_nonzero(self.status == edge_index + 1)
                 for edge_index in range(self.action_count)],
                dtype=np.float32,
            )
            self.previous_density = next_counts / self.areas
        else:
            self.previous_density.fill(0.0)
        self.current_step += 1

    def team_discounted_reward_to_go(self, gamma: float) -> np.ndarray:
        result = np.zeros(self.config.max_steps, dtype=np.float32)
        running = 0.0
        for step in range(self.current_step - 1, -1, -1):
            running = float(self.team_step_rewards[step]) + gamma * running
            result[step] = running
        return result

    def finished(self) -> bool:
        return self.current_step >= self.config.max_steps or np.all(
            self.status >= STATUS_ARRIVED
        )

    def finalize(self) -> dict[str, object]:
        timed_out = self.status < STATUS_ARRIVED
        self.status[timed_out] = STATUS_FAILED
        arrived = self.status == STATUS_ARRIVED
        timed_arrivals = np.where(arrived, self.travel_time, self.config.max_steps)
        chosen = self.action >= 0
        fractions = np.asarray(
            [np.mean(self.action[chosen] == edge) if np.any(chosen) else 0.0
             for edge in range(self.action_count)],
            dtype=float,
        )
        return {
            "mean_agent_reward": float(self.raw_returns.mean()),
            "total_reward": float(self.raw_returns.sum()),
            "arrival_count": int(arrived.sum()),
            "failure_count": int((self.status == STATUS_FAILED).sum()),
            "timeout_count": int(timed_out.sum()),
            "mean_arrival_time_arrived_only": (
                float(np.nanmean(self.travel_time[arrived]))
                if np.any(arrived) else float("nan")
            ),
            "mean_arrival_time_all_agents": float(timed_arrivals.mean()),
            "sampled_near_fraction": float(fractions[0]),
            "sampled_middle_fraction": float(fractions[1]),
            "sampled_far_fraction": float(fractions[2]),
            "per_agent_arrival_times": self.travel_time.copy(),
            "per_agent_rewards": self.raw_returns.copy(),
            "route_counts": np.bincount(
                self.action[chosen], minlength=self.action_count
            ).astype(int).tolist(),
        }
