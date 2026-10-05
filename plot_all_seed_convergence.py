"""Plot return and arrival-time convergence for all fixed-capacity training seeds."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37])
    parser.add_argument("--rolling-window", type=int, default=100)
    parser.add_argument("--summary-window", type=int, default=500)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def find_run(parent: Path, seed: int) -> Path:
    candidates = []
    for path in parent.rglob("diagnostics.pkl"):
        with path.open("rb") as handle:
            diagnostics = pickle.load(handle)
        if int(diagnostics.get("seed", -1)) == seed:
            candidates.append(path.parent)
    if not candidates:
        raise FileNotFoundError(f"No run found for seed {seed} below {parent}")
    return max(candidates, key=lambda path: (path / "diagnostics.pkl").stat().st_mtime)


def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    width = min(window, len(values))
    indices = np.arange(width - 1, len(values))
    smoothed = np.convolve(values, np.ones(width) / width, mode="valid")
    return indices, smoothed


def summarize(values: np.ndarray, width: int) -> dict[str, float]:
    width = min(width, len(values) // 2)
    previous = values[-2 * width:-width]
    final = values[-width:]
    return {
        "previous_mean": float(np.mean(previous)),
        "final_mean": float(np.mean(final)),
        "change": float(np.mean(final) - np.mean(previous)),
        "final_sd": float(np.std(final, ddof=1)),
    }


def main() -> None:
    args = parse_args()
    if args.rolling_window <= 0 or args.summary_window <= 0:
        raise ValueError("windows must be positive")

    runs = {}
    for seed in args.seeds:
        run = find_run(args.runs, seed)
        with (run / "diagnostics.pkl").open("rb") as handle:
            runs[seed] = (run, pickle.load(handle))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    colors = plt.cm.tab10(np.arange(len(args.seeds)))
    smooth_by_metric = {"return": [], "arrival": []}
    summaries = []

    for color, seed in zip(colors, args.seeds):
        run, diagnostics = runs[seed]
        episode = np.asarray(diagnostics["episode"], dtype=int)
        metrics = diagnostics["episode_metrics"]
        values_by_metric = {
            "return": np.asarray(metrics["mean_episode_return_per_agent"], dtype=float),
            "arrival": np.asarray(metrics["mean_arrival_time_with_timeouts"], dtype=float),
        }
        for ax, metric in ((axes[0, 0], "return"), (axes[0, 1], "arrival")):
            indices, smoothed = moving_average(
                values_by_metric[metric], args.rolling_window
            )
            x = episode[indices]
            ax.plot(x, smoothed, color=color, linewidth=1.35, alpha=0.9, label=f"seed {seed}")
            smooth_by_metric[metric].append((x, smoothed))

        validation = diagnostics.get("validation", [])
        vx = np.asarray([row["episode"] for row in validation], dtype=int)
        vr = np.asarray([row["mean_episode_return_per_agent"] for row in validation], dtype=float)
        vt = np.asarray([row["mean_arrival_time_with_timeouts"] for row in validation], dtype=float)
        axes[1, 0].plot(vx, vr, "o-", color=color, linewidth=1.1, markersize=4, label=f"seed {seed}")
        axes[1, 1].plot(vx, vt, "o-", color=color, linewidth=1.1, markersize=4, label=f"seed {seed}")

        summaries.append({
            "seed": seed,
            "run": str(run.resolve()),
            "episodes": int(episode[-1]),
            "return": summarize(values_by_metric["return"], args.summary_window),
            "arrival": summarize(values_by_metric["arrival"], args.summary_window),
        })

    for metric, ax in (("return", axes[0, 0]), ("arrival", axes[0, 1])):
        common_length = min(len(values) for _, values in smooth_by_metric[metric])
        common_x = smooth_by_metric[metric][0][0][-common_length:]
        stacked = np.vstack([
            values[-common_length:] for _, values in smooth_by_metric[metric]
        ])
        ax.plot(common_x, stacked.mean(axis=0), color="black", linewidth=2.4, label="4-seed mean")

    baseline = next(iter(runs.values()))[1].get("joint_fixed_baseline")
    if baseline:
        axes[1, 0].axhline(
            baseline["mean_episode_return_per_agent"], color="black",
            linestyle="--", linewidth=1.3, label="best fixed",
        )
        axes[1, 1].axhline(
            baseline["mean_arrival_time_with_timeouts"], color="black",
            linestyle="--", linewidth=1.3, label="best fixed",
        )

    titles = (
        "Training return — 100-episode moving average",
        "Training mean arrival time — 100-episode moving average",
        "Fixed-seed validation return",
        "Fixed-seed validation mean arrival time",
    )
    ylabels = ("Return", "Seconds", "Return", "Seconds")
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        ax.axvline(3000, color="0.45", linestyle="--", linewidth=0.9)
        ax.axvline(4000, color="0.45", linestyle=":", linewidth=0.9)
        ax.legend(ncol=2, fontsize=9)

    fig.suptitle(
        "Four-seed convergence — fixed near capacity 2200\n"
        "gray dashed: entropy annealing ends; gray dotted: original 4000-episode boundary",
        fontsize=16,
    )
    output = args.output or args.runs / "all_seed_return_arrival_convergence.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    summary = {
        "rolling_window": args.rolling_window,
        "summary_window": args.summary_window,
        "runs": summaries,
        "across_seed_final_return_mean": float(np.mean([
            row["return"]["final_mean"] for row in summaries
        ])),
        "across_seed_final_arrival_mean": float(np.mean([
            row["arrival"]["final_mean"] for row in summaries
        ])),
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for row in summaries:
        print(
            f"seed={row['seed']} final-{args.summary_window}: "
            f"return={row['return']['final_mean']:.3f} "
            f"(change={row['return']['change']:+.3f}), "
            f"arrival={row['arrival']['final_mean']:.2f}s "
            f"(change={row['arrival']['change']:+.2f}s)"
        )
    print(f"Saved: {output.resolve()}")
    print(f"Saved: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
