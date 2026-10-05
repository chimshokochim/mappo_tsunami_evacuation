"""Plot empirical convergence diagnostics for one resumed MAPPO training run."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--summary-window", type=int, default=500)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def rolling_mean(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    width = min(window, len(values))
    return (
        np.arange(width - 1, len(values)),
        np.convolve(values, np.ones(width) / width, mode="valid"),
    )


def window_summary(values: np.ndarray, width: int) -> dict[str, float]:
    width = min(width, len(values) // 2)
    previous = values[-2 * width:-width]
    final = values[-width:]
    x = np.arange(width, dtype=float)
    slope = float(np.polyfit(x, final, 1)[0] * 1000.0)
    return {
        "previous_mean": float(np.mean(previous)),
        "final_mean": float(np.mean(final)),
        "change": float(np.mean(final) - np.mean(previous)),
        "final_standard_deviation": float(np.std(final, ddof=1)),
        "final_slope_per_1000_episodes": slope,
    }


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    with (run / "diagnostics.pkl").open("rb") as handle:
        diagnostics = pickle.load(handle)
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))

    episode = np.asarray(diagnostics["episode"], dtype=int)
    episode_metrics = diagnostics["episode_metrics"]
    update_episode = np.asarray(diagnostics["update_episode"], dtype=int)
    update_metrics = diagnostics["update_metrics"]
    series = {
        "return": np.asarray(
            episode_metrics["mean_episode_return_per_agent"], dtype=float
        ),
        "arrival": np.asarray(
            episode_metrics["mean_arrival_time_with_timeouts"], dtype=float
        ),
        "far_fraction": np.asarray(
            episode_metrics["sampled_far_fraction"], dtype=float
        ),
        "failures": np.asarray(episode_metrics["failure_count"], dtype=float),
    }
    validation = diagnostics.get("validation", [])
    fixed = metadata.get("joint_fixed_baseline") or diagnostics.get(
        "joint_fixed_baseline"
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    plot_specs = [
        (axes[0, 0], "return", "Mean episode return / agent", "Return"),
        (axes[0, 1], "arrival", "Mean arrival time: all agents", "Seconds"),
        (axes[1, 0], "far_fraction", "Sampled far-choice fraction", "Fraction"),
        (axes[1, 1], "failures", "Failed agents", "Agents"),
    ]
    for ax, key, title, ylabel in plot_specs:
        values = series[key]
        ax.plot(episode, values, color="C0", alpha=0.12, linewidth=0.6, label="per episode")
        indices, smoothed = rolling_mean(values, args.window)
        ax.plot(
            episode[indices], smoothed, color="C0", linewidth=1.8,
            label=f"{args.window}-episode mean",
        )
        ax.set(title=title, xlabel="Episode", ylabel=ylabel)
        ax.grid(alpha=0.22)
        ax.axvline(3000, color="0.35", linestyle="--", linewidth=1.0)
        ax.axvline(4000, color="C3", linestyle=":", linewidth=1.2)
        ax.legend(loc="best")
    axes[1, 0].set_ylim(0.20, 0.48)
    axes[1, 1].set_ylim(-0.05, max(1.0, float(np.max(series["failures"])) + 0.2))

    entropy = np.asarray(update_metrics["policy_entropy"], dtype=float)
    coefficient = np.asarray(update_metrics["entropy_coefficient"], dtype=float)
    ax = axes[2, 0]
    ax.plot(update_episode, entropy, color="C2", linewidth=1.2, label="policy entropy")
    ax.set(
        title="Exploration schedule and policy entropy",
        xlabel="Episode", ylabel="Policy entropy",
    )
    ax.grid(alpha=0.22)
    ax.axhline(np.log(2), color="0.35", linestyle="--", linewidth=0.9, label="ln(2)")
    ax.axvline(3000, color="0.35", linestyle="--", linewidth=1.0)
    ax.axvline(4000, color="C3", linestyle=":", linewidth=1.2)
    ax2 = ax.twinx()
    ax2.plot(update_episode, coefficient, color="C1", linewidth=1.2, label="entropy coefficient")
    ax2.set_ylabel("Entropy coefficient")
    lines = [
        line for line in ax.get_lines() + ax2.get_lines()
        if not line.get_label().startswith("_")
    ]
    ax.legend(lines, [line.get_label() for line in lines], loc="best")

    ax = axes[2, 1]
    if validation:
        vx = np.asarray([row["episode"] for row in validation], dtype=int)
        vr = np.asarray([row["mean_episode_return_per_agent"] for row in validation])
        vt = np.asarray([row["mean_arrival_time_with_timeouts"] for row in validation])
        eligible = np.asarray([row.get("joint_eligible", False) for row in validation])
        ax.plot(vx, vr, "o-", color="C0", label="Actor validation return")
        if fixed:
            ax.axhline(
                fixed["mean_episode_return_per_agent"], color="C0", linestyle="--",
                linewidth=1.0, label="best-fixed return",
            )
        ax.scatter(vx[eligible], vr[eligible], s=80, facecolors="none", edgecolors="C2", linewidths=1.8, label="wins both")
        ax.set(xlabel="Episode", ylabel="Return")
        ax2 = ax.twinx()
        ax2.plot(vx, vt, "s-", color="C1", label="Actor validation arrival")
        if fixed:
            ax2.axhline(
                fixed["mean_arrival_time_with_timeouts"], color="C1", linestyle="--",
                linewidth=1.0, label="best-fixed arrival",
            )
        ax2.set_ylabel("Arrival time (seconds)")
        lines = ax.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        handles, labels2 = ax.get_legend_handles_labels()
        # Include the hollow eligibility marker as well as both axes' lines.
        ax.legend(lines + handles[-1:], labels + labels2[-1:], loc="best")
    ax.set_title("Fixed-seed validation checkpoints")
    ax.grid(alpha=0.22)

    fig.suptitle(
        "Seed 27 convergence diagnostics — fixed near capacity 2200, team gamma = 1.0\n"
        "annealing ends at episode 3000 (gray dashed); training resumes at 4000 (red dotted)",
        fontsize=16,
    )

    output = args.output or run / "seed27_convergence_diagnostics.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    summary = {
        "run": str(run),
        "rolling_window": args.window,
        "summary_window": args.summary_window,
        "return": window_summary(series["return"], args.summary_window),
        "arrival": window_summary(series["arrival"], args.summary_window),
        "far_fraction": window_summary(series["far_fraction"], args.summary_window),
        "failures": window_summary(series["failures"], args.summary_window),
        "last_validation": validation[-1] if validation else None,
        "joint_eligible_episodes": [
            row["episode"] for row in validation if row.get("joint_eligible", False)
        ],
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
