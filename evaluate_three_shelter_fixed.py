"""Sweep fixed near/middle/far probabilities in the three-shelter environment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from three_shelter_env import ThreeShelterConfig, ThreeShelterEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-agents", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--road-lengths", type=float, nargs=3, default=(150.0, 225.0, 300.0))
    parser.add_argument("--road-width", type=float, default=5.0)
    parser.add_argument("--probability-step", type=float, default=0.1)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=list(range(100, 110)))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_three_shelter_fixed"))
    return parser.parse_args()


def probability_grid(step: float) -> list[tuple[float, float, float]]:
    if not 0.0 < step <= 1.0:
        raise ValueError("probability-step must be in (0, 1]")
    intervals = round(1.0 / step)
    if not np.isclose(intervals * step, 1.0):
        raise ValueError("probability-step must divide 1.0 exactly")
    return [
        (near / intervals, middle / intervals, (intervals - near - middle) / intervals)
        for near in range(intervals + 1)
        for middle in range(intervals - near + 1)
    ]


def run_episode(
    config: ThreeShelterConfig,
    seed: int,
    probabilities: tuple[float, float, float],
    gamma: float = 0.99,
) -> dict[str, float]:
    env = ThreeShelterEnv(config, seed=seed)
    action_rng = np.random.default_rng(seed + 1_000_003)
    while not env.finished():
        ids, _, _ = env.decision_batch()
        if len(ids):
            actions = action_rng.choice(3, size=len(ids), p=probabilities)
            env.commit(ids, actions)
        env.advance(gamma)
    metrics = env.finalize()
    return {
        "mean_episode_return_per_agent": float(metrics["mean_agent_reward"]),
        "arrived_only_mean_arrival_time": float(metrics["mean_arrival_time_arrived_only"]),
        "mean_arrival_time_with_timeouts": float(metrics["mean_arrival_time_all_agents"]),
        "arrived_count": float(metrics["arrival_count"]),
        "failure_count": float(metrics["failure_count"]),
        "sampled_near_fraction": float(metrics["sampled_near_fraction"]),
        "sampled_middle_fraction": float(metrics["sampled_middle_fraction"]),
        "sampled_far_fraction": float(metrics["sampled_far_fraction"]),
    }


def aggregate(rows: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    keys = rows[0].keys()
    mean = {key: float(np.nanmean([row[key] for row in rows])) for key in keys}
    sd = {
        key: float(np.nanstd([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0
        for key in keys
    }
    return mean, sd


def main() -> None:
    args = parse_args()
    config = ThreeShelterConfig(
        num_agents=args.num_agents,
        max_steps=args.max_steps,
        road_lengths=tuple(args.road_lengths),
        road_width=args.road_width,
    )
    results = []
    for probabilities in probability_grid(args.probability_step):
        rows = [run_episode(config, seed, probabilities) for seed in args.evaluation_seeds]
        mean, sd = aggregate(rows)
        result = {
            "probability_near": probabilities[0],
            "probability_middle": probabilities[1],
            "probability_far": probabilities[2],
            "mean": mean,
            "sample_standard_deviation": sd,
        }
        results.append(result)
        print(
            f"p=({probabilities[0]:.2f},{probabilities[1]:.2f},{probabilities[2]:.2f}) "
            f"reward={mean['mean_episode_return_per_agent']:.3f} "
            f"arrival={mean['mean_arrival_time_with_timeouts']:.1f}s "
            f"failed={mean['failure_count']:.1f}",
            flush=True,
        )
    best = max(results, key=lambda row: row["mean"]["mean_episode_return_per_agent"])
    output = {
        "environment_config": asdict(config),
        "evaluation_seeds": args.evaluation_seeds,
        "probability_step": args.probability_step,
        "selection_rule": "maximum mean episode return per agent",
        "best_fixed_probability": best,
        "results": results,
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray([
        row["probability_middle"] + 0.5 * row["probability_far"] for row in results
    ])
    y = np.asarray([
        np.sqrt(3.0) * 0.5 * row["probability_far"] for row in results
    ])
    reward = np.asarray([row["mean"]["mean_episode_return_per_agent"] for row in results])
    arrival = np.asarray([row["mean"]["mean_arrival_time_with_timeouts"] for row in results])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for axis, values, title, label in (
        (axes[0], reward, "Mean episode return / agent", "Return"),
        (axes[1], arrival, "Mean arrival time: all agents", "Seconds"),
    ):
        image = axis.scatter(x, y, c=values, s=95, cmap="viridis")
        axis.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color="black", lw=1)
        axis.text(-0.03, -0.04, "Near", ha="right")
        axis.text(1.03, -0.04, "Middle", ha="left")
        axis.text(0.5, np.sqrt(3) / 2 + 0.035, "Far", ha="center")
        axis.set_aspect("equal")
        axis.set_axis_off()
        axis.set_title(title)
        fig.colorbar(image, ax=axis, label=label, shrink=0.8)
    fig.suptitle(
        f"Three-shelter fixed-policy sweep: {len(args.evaluation_seeds)} seeds per point"
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    json_path = args.output_dir / f"three_shelter_fixed_sweep_{stamp}.json"
    png_path = args.output_dir / f"three_shelter_fixed_sweep_{stamp}.png"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(
        "Best fixed probabilities: "
        f"near={best['probability_near']:.2f}, "
        f"middle={best['probability_middle']:.2f}, far={best['probability_far']:.2f}"
    )
    print(f"Saved: {png_path.resolve()}")
    print(f"Saved: {json_path.resolve()}")


if __name__ == "__main__":
    main()
