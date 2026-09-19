"""Plot training diagnostics from a three-shelter MAPPO run."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def resolve_diagnostics(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = list(path.rglob("diagnostics.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No diagnostics.pkl found under {path}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def moving_average(values, window: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return np.arange(1, len(values) + 1), values
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    return np.arange(window, len(values) + 1), smoothed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--episode-window", type=int, default=100)
    parser.add_argument("--update-window", type=int, default=25)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = resolve_diagnostics(args.run)
    with path.open("rb") as handle:
        data = pickle.load(handle)
    episode = data["episode_metrics"]
    update = data["update_metrics"]
    update_episode = np.asarray(data["update_episode"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    x, y = moving_average(episode["mean_episode_return_per_agent"], args.episode_window)
    axes[0, 0].plot(x, y)
    axes[0, 0].set_title("Mean episode return / agent")

    for key, label in (
        ("arrived_only_mean_arrival_time", "arrived only"),
        ("mean_arrival_time_with_timeouts", "all agents"),
    ):
        x, y = moving_average(episode[key], args.episode_window)
        axes[0, 1].plot(x, y, label=label)
    axes[0, 1].set_title("Mean arrival time")
    axes[0, 1].set_ylabel("Seconds")
    axes[0, 1].legend()

    for name in ("near", "middle", "far"):
        x, y = moving_average(episode[f"sampled_{name}_fraction"], args.episode_window)
        axes[0, 2].plot(x, y, label=name)
    axes[0, 2].set_title("Sampled shelter-choice fractions")
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].legend()

    for key, label in (("actor_loss", "Actor"), ("critic_loss", "Critic")):
        _, y = moving_average(update[key], args.update_window)
        x = update_episode[args.update_window - 1:] if len(update_episode) >= args.update_window else update_episode
        axes[1, 0].plot(x, y, label=label)
    axes[1, 0].set_title("PPO losses")
    axes[1, 0].legend()

    _, y = moving_average(update["policy_entropy"], args.update_window)
    x = update_episode[args.update_window - 1:] if len(update_episode) >= args.update_window else update_episode
    axes[1, 1].plot(x, y, label="policy entropy")
    axes[1, 1].axhline(np.log(3.0), color="black", ls="--", label="ln(3)")
    axes[1, 1].set_title("Categorical policy entropy")
    axes[1, 1].legend()

    for name in ("near", "middle", "far"):
        _, y = moving_average(update[f"mean_prob_{name}"], args.update_window)
        axes[1, 2].plot(x, y, label=name)
    axes[1, 2].set_title("Mean Actor probabilities")
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].legend()

    for axis in axes.flat:
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.25)
    fig.suptitle(f"Three-shelter MAPPO diagnostics — {path.parent.name}")
    output = args.output or path.with_name("three_shelter_training_diagnostics.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Loaded: {path.resolve()}")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
