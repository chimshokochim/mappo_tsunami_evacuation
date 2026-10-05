"""Compare when a trained Actor and a fixed policy send agents to far shelter."""

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


SUMMARY_METRICS = (
    "mean_episode_return_per_agent",
    "mean_arrival_time_with_timeouts",
    "sampled_far_fraction",
    "mean_actor_probability_far",
    "failure_count",
    "capacity_rejection_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actor",
        type=Path,
        required=True,
        help="actor.pt, its run directory, or a parent directory containing actor.pt",
    )
    parser.add_argument("--near-capacity", type=int, default=2200)
    parser.add_argument("--far-capacity", type=int, default=2400)
    parser.add_argument("--fixed-far-probability", type=float, default=0.30)
    parser.add_argument(
        "--evaluation-seeds", type=int, nargs="+", default=list(range(300, 320))
    )
    parser.add_argument(
        "--time-bin-seconds",
        type=float,
        default=10.0,
        help="width of time bins used in the figure",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("actor_fixed_timing_comparison")
    )
    return parser.parse_args()


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in SUMMARY_METRICS
    }


def trace_matrix(rows: list[dict], key: str, length: int) -> np.ndarray:
    matrix = np.full((len(rows), length), np.nan, dtype=np.float64)
    for row_index, row in enumerate(rows):
        values = np.asarray(row["trace"][key], dtype=np.float64)
        used = min(length, len(values))
        matrix[row_index, :used] = values[:used]
    return matrix


def bin_edges(length: int, bin_steps: int) -> list[tuple[int, int]]:
    return [(start, min(start + bin_steps, length)) for start in range(0, length, bin_steps)]


def bin_centres(length: int, bin_steps: int, dt: float) -> np.ndarray:
    return np.asarray(
        [0.5 * (start + end - 1) * dt for start, end in bin_edges(length, bin_steps)]
    )


def binned_matrix_mean(matrix: np.ndarray, bin_steps: int) -> np.ndarray:
    values = []
    for start, end in bin_edges(matrix.shape[1], bin_steps):
        block = matrix[:, start:end]
        finite = block[np.isfinite(block)]
        values.append(float(finite.mean()) if len(finite) else np.nan)
    return np.asarray(values)


def summed_step_trace(rows: list[dict], key: str, length: int) -> np.ndarray:
    total = np.zeros(length, dtype=np.float64)
    for row in rows:
        values = np.asarray(row["trace"][key], dtype=np.float64)
        total[: min(length, len(values))] += values[:length]
    return total


def binned_ratio(
    numerator: np.ndarray, denominator: np.ndarray, bin_steps: int
) -> np.ndarray:
    ratios = []
    for start, end in bin_edges(len(denominator), bin_steps):
        denominator_sum = float(denominator[start:end].sum())
        ratios.append(
            float(numerator[start:end].sum()) / denominator_sum
            if denominator_sum > 0.0
            else np.nan
        )
    return np.asarray(ratios)


def cumulative_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator_cumulative = np.cumsum(numerator)
    denominator_cumulative = np.cumsum(denominator)
    return np.divide(
        numerator_cumulative,
        denominator_cumulative,
        out=np.full_like(numerator_cumulative, np.nan, dtype=np.float64),
        where=denominator_cumulative > 0.0,
    )


def add_departure_end(ax: plt.Axes, departure_end: float) -> None:
    ax.axvline(
        departure_end,
        color="0.35",
        linestyle=":",
        linewidth=1.2,
        label="departure-window end",
    )


def main() -> None:
    args = parse_args()
    if args.near_capacity < 0 or args.far_capacity < 0:
        raise ValueError("Shelter capacities must be non-negative")
    if not 0.0 <= args.fixed_far_probability <= 1.0:
        raise ValueError("--fixed-far-probability must be in [0, 1]")
    if args.time_bin_seconds <= 0.0:
        raise ValueError("--time-bin-seconds must be positive")
    if len(set(args.evaluation_seeds)) != len(args.evaluation_seeds):
        raise ValueError("--evaluation-seeds contains duplicates")

    actor, training_config, actor_path = load_actor_and_config(args.actor)
    if args.near_capacity + args.far_capacity < training_config.num_agents:
        raise ValueError("The two shelters cannot accommodate all agents")
    config = replace(
        training_config,
        near_shelter_capacity=args.near_capacity,
        near_shelter_capacity_candidates=(),
        far_shelter_capacity=args.far_capacity,
        shelter_capacity_mode="reject",
        closure_edge=None,
        random_blockage_probability=0.0,
    )

    actor_rows = []
    fixed_rows = []
    for index, seed in enumerate(args.evaluation_seeds, start=1):
        actor_rows.append(
            run_evaluation_episode(config, seed, actor=actor, collect_trace=True)
        )
        fixed_rows.append(
            run_evaluation_episode(
                config,
                seed,
                fixed_far_probability=args.fixed_far_probability,
                collect_trace=True,
            )
        )
        print(f"evaluation seed {seed} complete ({index}/{len(args.evaluation_seeds)})")

    departure_steps = max(
        1, int(np.ceil(config.max_steps * config.departure_window_fraction))
    )
    trace_steps = max(
        max(len(row["trace"]["physical_density_near"]) for row in actor_rows),
        max(len(row["trace"]["physical_density_near"]) for row in fixed_rows),
    )
    bin_steps = max(1, int(round(args.time_bin_seconds / config.dt)))
    decision_time = bin_centres(departure_steps, bin_steps, config.dt)
    density_time = bin_centres(trace_steps, bin_steps, config.dt)
    step_time = np.arange(departure_steps, dtype=np.float64) * config.dt
    departure_end = departure_steps * config.dt

    series = {}
    for label, rows in (("actor", actor_rows), ("fixed", fixed_rows)):
        decisions = summed_step_trace(rows, "decision_count", departure_steps)
        far_decisions = summed_step_trace(rows, "far_decisions", departure_steps)
        probability_sum = summed_step_trace(rows, "far_probability_sum", departure_steps)
        series[label] = {
            "probability_far": binned_ratio(probability_sum, decisions, bin_steps),
            "actual_far": binned_ratio(far_decisions, decisions, bin_steps),
            "cumulative_far": cumulative_ratio(far_decisions, decisions),
            "near_density": binned_matrix_mean(
                trace_matrix(rows, "physical_density_near", trace_steps), bin_steps
            ),
            "far_density": binned_matrix_mean(
                trace_matrix(rows, "physical_density_far", trace_steps), bin_steps
            ),
            "remaining_near_capacity": binned_matrix_mean(
                trace_matrix(rows, "remaining_near_capacity", departure_steps),
                bin_steps,
            ),
            "waiting_agents": binned_matrix_mean(
                trace_matrix(rows, "waiting_agents", departure_steps), bin_steps
            ),
        }

    actor_colour = "tab:blue"
    fixed_colour = "tab:orange"
    fig, axes = plt.subplots(3, 2, figsize=(16, 13), constrained_layout=True)
    fig.suptitle(
        "When does the trained Actor send agents to the far shelter?\n"
        f"near capacity={args.near_capacity}, far capacity={args.far_capacity}, "
        f"{len(args.evaluation_seeds)} evaluation seed(s)",
        fontsize=16,
    )

    ax = axes[0, 0]
    ax.plot(decision_time, series["actor"]["probability_far"], color=actor_colour,
            linewidth=2.2, label="Actor mean P(far)")
    ax.plot(decision_time, series["actor"]["actual_far"], color=actor_colour,
            linestyle="--", label="Actor actual far fraction")
    ax.plot(decision_time, series["fixed"]["actual_far"], color=fixed_colour,
            linewidth=2.0, label=f"fixed actual far fraction (p={args.fixed_far_probability:.2f})")
    ax.set_title("Far choice within each time bin")
    ax.set_ylabel("Probability / fraction")
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(step_time, series["actor"]["cumulative_far"], color=actor_colour,
            linewidth=2.2, label="Actor")
    ax.plot(step_time, series["fixed"]["cumulative_far"], color=fixed_colour,
            linewidth=2.2, label=f"fixed p={args.fixed_far_probability:.2f}")
    ax.axhline(args.fixed_far_probability, color=fixed_colour, linestyle=":", alpha=0.8)
    ax.set_title("Cumulative far-choice fraction")
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0.0, 0.6)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(density_time, series["actor"]["near_density"], color=actor_colour,
            linewidth=2.0, label="Actor")
    ax.plot(density_time, series["fixed"]["near_density"], color=fixed_colour,
            linewidth=2.0, label=f"fixed p={args.fixed_far_probability:.2f}")
    add_departure_end(ax, departure_end)
    ax.set_title("Near-edge density")
    ax.set_ylabel("Agents / m²")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(density_time, series["actor"]["far_density"], color=actor_colour,
            linewidth=2.0, label="Actor")
    ax.plot(density_time, series["fixed"]["far_density"], color=fixed_colour,
            linewidth=2.0, label=f"fixed p={args.fixed_far_probability:.2f}")
    add_departure_end(ax, departure_end)
    ax.set_title("Far-edge density")
    ax.set_ylabel("Agents / m²")
    ax.legend()

    ax = axes[2, 0]
    ax.plot(decision_time, series["actor"]["remaining_near_capacity"],
            color=actor_colour, linewidth=2.2, label="Actor")
    ax.plot(decision_time, series["fixed"]["remaining_near_capacity"],
            color=fixed_colour, linewidth=2.2, label=f"fixed p={args.fixed_far_probability:.2f}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Remaining near-shelter capacity")
    ax.set_ylabel("Spaces")
    ax.legend()

    ax = axes[2, 1]
    ax.plot(decision_time, series["actor"]["waiting_agents"], color=actor_colour,
            linewidth=2.2, label="Actor")
    ax.plot(decision_time, series["fixed"]["waiting_agents"], color=fixed_colour,
            linewidth=2.2, label=f"fixed p={args.fixed_far_probability:.2f}")
    ax.set_title("Agents still waiting to depart")
    ax.set_ylabel("Agents")
    ax.legend()

    for ax in axes.flat:
        ax.set_xlabel("Time (seconds)")
        ax.grid(alpha=0.25)

    actor_summary = mean_metrics(actor_rows)
    fixed_summary = mean_metrics(fixed_rows)
    result = {
        "actor_path": str(actor_path.resolve()),
        "evaluation_seeds": args.evaluation_seeds,
        "near_capacity": args.near_capacity,
        "far_capacity": args.far_capacity,
        "fixed_far_probability": args.fixed_far_probability,
        "time_bin_seconds": bin_steps * config.dt,
        "actor_mean": actor_summary,
        "fixed_mean": fixed_summary,
        "fixed_minus_actor_return": (
            fixed_summary["mean_episode_return_per_agent"]
            - actor_summary["mean_episode_return_per_agent"]
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    figure_path = args.output_dir / f"actor_vs_fixed_timing_{timestamp}.png"
    json_path = args.output_dir / f"actor_vs_fixed_timing_{timestamp}.json"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        "Actor: "
        f"return={actor_summary['mean_episode_return_per_agent']:.3f}, "
        f"arrival={actor_summary['mean_arrival_time_with_timeouts']:.1f}s, "
        f"far={actor_summary['sampled_far_fraction']:.3f}, "
        f"failures={actor_summary['failure_count']:.2f}"
    )
    print(
        "Fixed: "
        f"return={fixed_summary['mean_episode_return_per_agent']:.3f}, "
        f"arrival={fixed_summary['mean_arrival_time_with_timeouts']:.1f}s, "
        f"far={fixed_summary['sampled_far_fraction']:.3f}, "
        f"failures={fixed_summary['failure_count']:.2f}"
    )
    print(f"Saved: {figure_path.resolve()}")
    print(f"Saved: {json_path.resolve()}")


if __name__ == "__main__":
    main()
