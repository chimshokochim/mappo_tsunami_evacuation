"""Compare gamma=0.99 and gamma=1.0 team-return Actors with fixed p(far)."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation_utils import load_actor_and_config, run_evaluation_episode


METRICS = (
    "mean_episode_return_per_agent",
    "mean_arrival_time_with_timeouts",
    "sampled_far_fraction",
    "failure_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma099-actor", type=Path, required=True)
    parser.add_argument("--gamma1-actor", type=Path, required=True)
    parser.add_argument("--near-capacity", type=int, default=2200)
    parser.add_argument("--far-capacity", type=int, default=2400)
    parser.add_argument("--fixed-far-probability", type=float, default=0.30)
    parser.add_argument(
        "--evaluation-seeds", type=int, nargs="+", default=list(range(300, 320))
    )
    parser.add_argument("--time-bin-seconds", type=float, default=10.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("team_return_gamma_comparison")
    )
    return parser.parse_args()


def aggregate(rows: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    mean = {key: float(np.mean([row[key] for row in rows])) for key in METRICS}
    sd = {
        key: float(np.std([row[key] for row in rows], ddof=1))
        if len(rows) > 1 else 0.0
        for key in METRICS
    }
    return mean, sd


def summed_trace(rows: list[dict], key: str, length: int) -> np.ndarray:
    result = np.zeros(length, dtype=np.float64)
    for row in rows:
        values = np.asarray(row["trace"][key], dtype=np.float64)
        used = min(length, len(values))
        result[:used] += values[:used]
    return result


def binned_ratio(
    numerator: np.ndarray, denominator: np.ndarray, bin_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    x_values = []
    y_values = []
    for start in range(0, len(denominator), bin_steps):
        end = min(start + bin_steps, len(denominator))
        count = float(denominator[start:end].sum())
        x_values.append(0.5 * (start + end - 1))
        y_values.append(
            float(numerator[start:end].sum()) / count if count > 0.0 else np.nan
        )
    return np.asarray(x_values), np.asarray(y_values)


def cumulative_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.cumsum(numerator)
    denominator = np.cumsum(denominator)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.fixed_far_probability <= 1.0:
        raise ValueError("--fixed-far-probability must be in [0, 1]")
    if args.time_bin_seconds <= 0.0:
        raise ValueError("--time-bin-seconds must be positive")
    if len(set(args.evaluation_seeds)) != len(args.evaluation_seeds):
        raise ValueError("--evaluation-seeds contains duplicates")

    actor_099, config_099, path_099 = load_actor_and_config(args.gamma099_actor)
    actor_1, config_1, path_1 = load_actor_and_config(args.gamma1_actor)
    if (
        config_099.num_agents != config_1.num_agents
        or actor_099.input_dim != actor_1.input_dim
    ):
        raise ValueError("The two Actor runs are not directly comparable")
    if args.near_capacity + args.far_capacity < config_1.num_agents:
        raise ValueError("The two shelters cannot accommodate all agents")

    config = replace(
        config_1,
        near_shelter_capacity=args.near_capacity,
        near_shelter_capacity_candidates=(),
        far_shelter_capacity=args.far_capacity,
        shelter_capacity_mode="reject",
        closure_edge=None,
        random_blockage_probability=0.0,
    )
    policies = (
        ("team gamma=0.99", actor_099, None),
        ("team gamma=1.0", actor_1, None),
        (f"fixed p={args.fixed_far_probability:.2f}", None, args.fixed_far_probability),
    )
    rows_by_policy: dict[str, list[dict]] = {}
    for label, actor, fixed_probability in policies:
        rows = []
        for index, seed in enumerate(args.evaluation_seeds, start=1):
            rows.append(
                run_evaluation_episode(
                    config,
                    seed,
                    actor=actor,
                    fixed_far_probability=fixed_probability,
                    collect_trace=True,
                )
            )
            print(f"{label}: seed {seed} complete ({index}/{len(args.evaluation_seeds)})")
        rows_by_policy[label] = rows

    labels = [item[0] for item in policies]
    colours = ["tab:gray", "tab:blue", "tab:orange"]
    means = {}
    sds = {}
    for label in labels:
        means[label], sds[label] = aggregate(rows_by_policy[label])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(
        "Effect of removing per-second discount from team reward-to-go\n"
        f"near capacity={args.near_capacity}, far capacity={args.far_capacity}, "
        f"{len(args.evaluation_seeds)} evaluation seeds",
        fontsize=16,
    )
    bar_specs = (
        ("mean_episode_return_per_agent", "Mean episode return / agent", "Return"),
        ("mean_arrival_time_with_timeouts", "Mean arrival time", "Seconds"),
        ("sampled_far_fraction", "Actual far-choice fraction", "Fraction"),
        ("failure_count", "Failed agents", "Agents"),
    )
    for ax, (key, title, ylabel) in zip(axes.flat[:4], bar_specs):
        positions = np.arange(len(labels))
        values = [means[label][key] for label in labels]
        errors = [sds[label][key] for label in labels]
        for position, value, error, colour in zip(
            positions, values, errors, colours
        ):
            ax.errorbar(
                position, value, yerr=error, fmt="o", color=colour,
                markersize=9, capsize=5, linewidth=2,
            )
        ax.set_xticks(positions, labels, rotation=12)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        if key == "failure_count" and max(values) == min(values) == 0.0:
            ax.set_ylim(-0.05, 0.05)

    departure_steps = max(
        1, int(np.ceil(config.max_steps * config.departure_window_fraction))
    )
    bin_steps = max(1, int(round(args.time_bin_seconds / config.dt)))
    time = np.arange(departure_steps, dtype=np.float64) * config.dt

    ax_probability = axes[1, 1]
    ax_cumulative = axes[1, 2]
    for label, colour in zip(labels, colours):
        rows = rows_by_policy[label]
        counts = summed_trace(rows, "decision_count", departure_steps)
        probability_sum = summed_trace(rows, "far_probability_sum", departure_steps)
        far_decisions = summed_trace(rows, "far_decisions", departure_steps)
        bin_x, bin_probability = binned_ratio(probability_sum, counts, bin_steps)
        ax_probability.plot(
            bin_x * config.dt, bin_probability, color=colour, linewidth=2.2, label=label
        )
        ax_cumulative.plot(
            time,
            cumulative_ratio(far_decisions, counts),
            color=colour,
            linewidth=2.2,
            label=label,
        )

    ax_probability.set_title("P(far) by departure time")
    ax_probability.set_xlabel("Time (seconds)")
    ax_probability.set_ylabel("Mean probability")
    ax_probability.set_ylim(0.0, 1.0)
    ax_probability.grid(alpha=0.25)
    ax_probability.legend()

    ax_cumulative.set_title("Cumulative actual far-choice fraction")
    ax_cumulative.set_xlabel("Time (seconds)")
    ax_cumulative.set_ylabel("Fraction")
    ax_cumulative.set_ylim(0.0, 1.0)
    ax_cumulative.grid(alpha=0.25)
    ax_cumulative.legend()

    result = {
        "gamma099_actor": str(path_099.resolve()),
        "gamma1_actor": str(path_1.resolve()),
        "evaluation_seeds": args.evaluation_seeds,
        "near_capacity": args.near_capacity,
        "far_capacity": args.far_capacity,
        "fixed_far_probability": args.fixed_far_probability,
        "error_bars": "sample SD across evaluation episodes",
        "policy_means": means,
        "policy_standard_deviations": sds,
        "per_seed_metrics": {
            label: [
                {"seed": row["seed"], **{key: float(row[key]) for key in METRICS}}
                for row in rows_by_policy[label]
            ]
            for label in labels
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    figure_path = args.output_dir / f"team_return_gamma_comparison_{timestamp}.png"
    json_path = args.output_dir / f"team_return_gamma_comparison_{timestamp}.json"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for label in labels:
        print(
            f"{label}: return={means[label]['mean_episode_return_per_agent']:.3f}, "
            f"arrival={means[label]['mean_arrival_time_with_timeouts']:.1f}s, "
            f"far={means[label]['sampled_far_fraction']:.3f}, "
            f"failures={means[label]['failure_count']:.2f}"
        )
    print(f"Saved: {figure_path.resolve()}")
    print(f"Saved: {json_path.resolve()}")


if __name__ == "__main__":
    main()
