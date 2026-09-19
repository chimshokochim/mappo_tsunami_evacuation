"""Diagnostic tests for the capacity-free three-shelter extension."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from three_shelter_env import ThreeShelterConfig, ThreeShelterEnv
from three_shelter_training import (
    Actor,
    Critic,
    MAPPOAgent,
    PPOConfig,
    RolloutBuffer,
    collect_episode,
    restore_checkpoint,
    save_checkpoint,
)


class ThreeShelterTests(unittest.TestCase):
    def test_network_shapes_and_probabilities(self):
        actor = Actor()
        critic = Critic()
        obs = torch.randn(11, 4)
        state = torch.randn(11, 7)
        probabilities = actor(obs)
        self.assertEqual(tuple(probabilities.shape), (11, 3))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(11), atol=1e-6))
        self.assertEqual(tuple(critic(state).shape), (11,))

    def test_environment_observation_and_actions(self):
        env = ThreeShelterEnv(ThreeShelterConfig(num_agents=12), seed=5)
        ids = np.asarray([0, 1, 2], dtype=np.int64)
        env.start_steps[ids] = 0
        batch_ids, obs, state = env.decision_batch()
        selected = np.isin(batch_ids, ids)
        self.assertEqual(obs.shape[1], 4)
        self.assertEqual(state.shape[1], 7)
        self.assertEqual(tuple(env.action_mask().shape), (3,))
        chosen_ids = batch_ids[selected]
        env.commit(chosen_ids, np.asarray([0, 1, 2], dtype=np.int64))
        self.assertEqual(env.action[chosen_ids].tolist(), [0, 1, 2])

    def test_rollout_requires_four_complete_episodes(self):
        buffer = RolloutBuffer(4)
        episode = {"x": np.arange(3)}
        for _ in range(3):
            buffer.add(episode)
            self.assertFalse(buffer.ready)
        buffer.add(episode)
        self.assertTrue(buffer.ready)
        self.assertEqual(len(buffer.consume()["x"]), 12)
        self.assertEqual(buffer.episode_count, 0)

    def test_learning_rates_and_entropy_schedule(self):
        config = PPOConfig(entropy_mode="annealed")
        agent = MAPPOAgent(config, torch.device("cpu"))
        self.assertEqual(agent.actor_optimizer.param_groups[0]["lr"], 1e-4)
        self.assertEqual(agent.critic_optimizer.param_groups[0]["lr"], 1e-3)
        self.assertAlmostEqual(agent.entropy_coefficient(0), 0.05)
        self.assertAlmostEqual(agent.entropy_coefficient(4500), 0.03)
        self.assertAlmostEqual(agent.entropy_coefficient(9000), 0.01)
        self.assertAlmostEqual(agent.entropy_coefficient(12000), 0.01)

    def test_episode_has_one_transition_per_agent(self):
        config = ThreeShelterConfig(num_agents=120, max_steps=900)
        env = ThreeShelterEnv(config, seed=9)
        agent = MAPPOAgent(PPOConfig(minibatch_size=64), torch.device("cpu"))
        batch, metrics = collect_episode(env, agent)
        self.assertEqual(len(batch["actions"]), 120)
        self.assertEqual(batch["observations"].shape, (120, 4))
        self.assertEqual(batch["states"].shape, (120, 7))
        total_fraction = sum(metrics[f"sampled_{name}_fraction"] for name in ("near", "middle", "far"))
        self.assertAlmostEqual(total_fraction, 1.0, places=6)
        for value in batch.values():
            self.assertTrue(np.all(np.isfinite(value)))

    def test_update_changes_actor_and_is_finite(self):
        torch.manual_seed(3)
        rng = np.random.default_rng(3)
        config = PPOConfig(minibatch_size=32)
        agent = MAPPOAgent(config, torch.device("cpu"))
        observations = rng.normal(size=(96, 4)).astype(np.float32)
        states = rng.normal(size=(96, 7)).astype(np.float32)
        with torch.no_grad():
            probs = agent.actor(torch.from_numpy(observations))
            actions = torch.multinomial(probs, 1).squeeze(1)
            old_log_probs = torch.log(
                probs.gather(1, actions[:, None]).squeeze(1)
            )
        batch = {
            "observations": observations,
            "states": states,
            "actions": actions.numpy(),
            "log_probs": old_log_probs.numpy(),
            "returns": rng.normal(size=96).astype(np.float32),
        }
        before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
        metrics = agent.update(batch, entropy_coef=0.05)
        self.assertTrue(any(
            not torch.equal(old, new.detach())
            for old, new in zip(before, agent.actor.parameters())
        ))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertEqual(metrics["transition_count"], 96)

    def test_single_atomic_checkpoint_can_restore(self):
        config = PPOConfig(minibatch_size=32)
        env = ThreeShelterEnv(ThreeShelterConfig(num_agents=24), seed=12)
        agent = MAPPOAgent(config, torch.device("cpu"))
        agent.update_count = 3
        diagnostics = {"episode": [1, 2, 3, 4], "marker": "first"}
        expected = [parameter.detach().clone() for parameter in agent.actor.parameters()]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_latest.pt"
            save_checkpoint(
                path, next_episode=4, target_episodes=8, env=env,
                agent=agent, diagnostics=diagnostics, seed=12,
            )
            diagnostics["marker"] = "replacement"
            save_checkpoint(
                path, next_episode=4, target_episodes=8, env=env,
                agent=agent, diagnostics=diagnostics, seed=12,
            )
            self.assertEqual([item.name for item in Path(directory).iterdir()], [path.name])
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["diagnostics"]["marker"], "replacement")
            restored_env = ThreeShelterEnv(ThreeShelterConfig(num_agents=24), seed=99)
            restored_agent = MAPPOAgent(config, torch.device("cpu"))
            restore_checkpoint(checkpoint, restored_env, restored_agent)
            self.assertEqual(restored_agent.update_count, 3)
            for old, new in zip(expected, restored_agent.actor.parameters()):
                self.assertTrue(torch.equal(old, new.detach()))


if __name__ == "__main__":
    unittest.main()
