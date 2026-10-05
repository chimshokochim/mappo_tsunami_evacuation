import numpy as np
import torch

from evac_env import EvacuationConfig, EvacuationEnv
from training import (
    MAPPOAgent,
    PPOConfig,
    balanced_episode_condition,
    build_mixed_demand_conditions,
    collect_episode,
)


def main() -> None:
    conditions = build_mixed_demand_conditions(
        (1000, 2000, 3000, 4000, 5000), (0.6, 0.7, 0.8), 0.8
    )
    assert len(conditions) == 15
    assert set(
        balanced_episode_condition(conditions, episode, 7)
        for episode in range(15)
    ) == set(conditions)

    config = EvacuationConfig(
        num_agents=1000,
        max_steps=20,
        departure_window_fraction=0.1,
        near_length=0.1,
        far_length=0.1,
        near_shelter_capacity=600,
        far_shelter_capacity=800,
        shelter_capacity_mode="reject",
        observation_schema="capacity_absolute_and_ratio",
        include_total_demand_observation=True,
        total_demand_observation_scale=5000,
    )
    env = EvacuationEnv(config, seed=7)
    env.reset(seed=7, episode_condition=(2000, 1400, 1600))
    ids, observations, states = env.decision_batch()
    assert observations.shape[1] == 8
    assert states.shape[1] == 10
    assert np.allclose(observations[:, -1], 0.4)
    assert np.allclose(states[:, -1], 0.4)

    agent = MAPPOAgent(
        PPOConfig(observation_dim=8, critic_state_dim=10),
        torch.device("cpu"),
    )
    batch, metrics = collect_episode(
        env, agent, reset_seed=11,
        episode_condition=(1000, 800, 800),
    )
    assert batch["observations"].shape == (1000, 8)
    assert batch["states"].shape == (1000, 10)
    assert metrics["total_count"] == 1000
    print("Mixed-demand smoke test passed")


if __name__ == "__main__":
    main()
