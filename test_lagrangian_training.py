import numpy as np
import torch

from training import MAPPOAgent, PPOConfig


def _synthetic_batch(size: int, obs_dim: int, state_dim: int, num_costs: int):
    rng = np.random.default_rng(123)
    probabilities = np.full((size, 2), 0.5, dtype=np.float32)
    actions = rng.integers(0, 2, size=size, dtype=np.int64)
    return {
        "observations": rng.normal(size=(size, obs_dim)).astype(np.float32),
        "states": rng.normal(size=(size, state_dim)).astype(np.float32),
        "actions": actions,
        "action_masks": np.ones((size, 2), dtype=np.float32),
        "log_probs": np.log(probabilities[np.arange(size), actions]),
        "returns": rng.normal(size=size).astype(np.float32),
        "cost_returns": np.column_stack((
            np.array([0, 1, 0, 1], dtype=np.float32),
            np.array([0, 0, 0, 1], dtype=np.float32),
        ))[:, :num_costs],
    }


def main() -> None:
    torch.manual_seed(3)
    baseline = MAPPOAgent(PPOConfig(), torch.device("cpu"))
    assert baseline.cost_critic is None
    assert baseline.lagrange_multipliers is None

    config = PPOConfig(
        observation_dim=3,
        critic_state_dim=4,
        constraint_mode="lagrangian",
        num_costs=2,
        cost_limits=(0.1, 0.5),
        lagrange_learning_rates=(0.1, 0.2),
        lagrange_initial_values=(0.0, 0.0),
        ppo_epochs=1,
        minibatch_size=4,
    )
    agent = MAPPOAgent(config, torch.device("cpu"))
    metrics = agent.update(_synthetic_batch(4, 3, 4, 2), entropy_coef=0.0)

    # Observed costs are (0.5, 0.25), so violations are (0.4, -0.25).
    # Projection onto the non-negative orthant gives lambda=(0.04, 0.0).
    assert np.isclose(metrics["observed_cost_0"], 0.5)
    assert np.isclose(metrics["observed_cost_1"], 0.25)
    assert np.isclose(metrics["constraint_violation_0"], 0.4)
    assert np.isclose(metrics["constraint_violation_1"], -0.25)
    assert np.isclose(metrics["lagrange_multiplier_0"], 0.04)
    assert np.isclose(metrics["lagrange_multiplier_1"], 0.0)
    assert np.isfinite(metrics["cost_critic_loss"])
    assert agent.cost_critic is not None
    print("Lagrangian PPO unit test passed")


if __name__ == "__main__":
    main()
