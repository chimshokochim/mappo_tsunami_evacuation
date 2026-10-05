"""Compare convergence episode and wall time across minibatch configurations."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", action="append", required=True,
        help="configuration in LABEL=RUN_PARENT form; repeat for each setting",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37])
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--check-interval", type=int, default=200)
    parser.add_argument("--min-episode", type=int, default=2000)
    parser.add_argument("--required-consecutive", type=int, default=3)
    parser.add_argument("--return-tolerance", type=float, default=0.5)
    parser.add_argument("--arrival-tolerance", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_configs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        if not label or label in result:
            raise ValueError(f"Invalid or duplicate label: {label!r}")
        result[label] = Path(raw_path)
    return result


def discover(parent: Path, seeds: list[int]) -> dict[int, tuple[Path, dict]]:
    selected = {}
    for path in parent.rglob("diagnostics.pkl"):
        with path.open("rb") as handle:
            diagnostics = pickle.load(handle)
        seed = int(diagnostics.get("seed", -1))
        if seed not in seeds:
            continue
        previous = selected.get(seed)
        if previous is None or path.stat().st_mtime > previous[0].stat().st_mtime:
            selected[seed] = (path, diagnostics)
    missing = [seed for seed in seeds if seed not in selected]
    if missing:
        raise FileNotFoundError(f"Missing seeds {missing} below {parent}")
    return selected


def detect_convergence(
    diagnostics: dict, window: int, interval: int, min_episode: int,
    required: int, return_tolerance: float, arrival_tolerance: float,
) -> tuple[int | None, list[dict]]:
    metrics = diagnostics["episode_metrics"]
    returns = np.asarray(metrics["mean_episode_return_per_agent"], dtype=float)
    arrivals = np.asarray(metrics["mean_arrival_time_with_timeouts"], dtype=float)
    failures = np.asarray(metrics["failure_count"], dtype=float)
    maximum = len(returns)
    first_check = max(min_episode, 2 * window)
    first_check = ((first_check + interval - 1) // interval) * interval
    streak = 0
    rows = []
    converged = None
    for episode in range(first_check, maximum + 1, interval):
        previous = slice(episode - 2 * window, episode - window)
        recent = slice(episode - window, episode)
        return_change = float(returns[recent].mean() - returns[previous].mean())
        arrival_change = float(arrivals[recent].mean() - arrivals[previous].mean())
        stable = (
            abs(return_change) <= return_tolerance
            and abs(arrival_change) <= arrival_tolerance
            and float(failures[recent].max()) == 0.0
        )
        streak = streak + 1 if stable else 0
        rows.append({
            "episode": episode,
            "return_change": return_change,
            "arrival_change": arrival_change,
            "max_failures": float(failures[recent].max()),
            "stable": stable,
            "consecutive": streak,
        })
        if converged is None and streak >= required:
            converged = episode
    return converged, rows


def elapsed_at_episode(diagnostics: dict, episode: int | None) -> float | None:
    if episode is None:
        return None
    update_episodes = np.asarray(diagnostics["update_episode"], dtype=int)
    elapsed = np.asarray(diagnostics["runtime"]["elapsed_seconds"], dtype=float)
    if len(update_episodes) != len(elapsed) or len(elapsed) == 0:
        return None
    # Resumed runs restart elapsed_seconds, so only report time when monotonic.
    if np.any(np.diff(elapsed) < 0):
        return None
    index = int(np.searchsorted(update_episodes, episode, side="left"))
    index = min(index, len(elapsed) - 1)
    return float(elapsed[index])


def moving_average(values: np.ndarray, width: int = 100) -> np.ndarray:
    width = min(width, len(values))
    return np.convolve(values, np.ones(width) / width, mode="valid")


def main() -> None:
    args = parse_args()
    configs = parse_configs(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "criterion": {
            "window": args.window,
            "check_interval": args.check_interval,
            "min_episode": args.min_episode,
            "required_consecutive": args.required_consecutive,
            "return_tolerance": args.return_tolerance,
            "arrival_tolerance_seconds": args.arrival_tolerance,
            "failure_requirement": "maximum failures in recent window equals zero",
        },
        "configurations": [],
    }
    plot_data = {}

    for label, parent in configs.items():
        runs = discover(parent, args.seeds)
        rows = []
        return_curves = []
        arrival_curves = []
        for seed in args.seeds:
            path, diagnostics = runs[seed]
            convergence, checks = detect_convergence(
                diagnostics, args.window, args.check_interval, args.min_episode,
                args.required_consecutive, args.return_tolerance,
                args.arrival_tolerance,
            )
            metrics = diagnostics["episode_metrics"]
            returns = np.asarray(metrics["mean_episode_return_per_agent"], dtype=float)
            arrivals = np.asarray(metrics["mean_arrival_time_with_timeouts"], dtype=float)
            return_curves.append(moving_average(returns))
            arrival_curves.append(moving_average(arrivals))
            metadata = json.loads((path.parent / "run_metadata.json").read_text(encoding="utf-8"))
            rows.append({
                "seed": seed,
                "run": str(path.parent.resolve()),
                "episodes": len(returns),
                "convergence_episode": convergence,
                "wall_seconds_to_convergence": elapsed_at_episode(diagnostics, convergence),
                "final_500_return": float(returns[-500:].mean()),
                "final_500_arrival": float(arrivals[-500:].mean()),
                "has_robust_checkpoint": bool(metadata.get("has_best_robust_joint_actor", False)),
                "checks": checks,
            })
        convergence_values = [row["convergence_episode"] for row in rows if row["convergence_episode"] is not None]
        wall_values = [row["wall_seconds_to_convergence"] for row in rows if row["wall_seconds_to_convergence"] is not None]
        aggregate = {
            "converged_seed_count": len(convergence_values),
            "robust_checkpoint_count": sum(row["has_robust_checkpoint"] for row in rows),
            "mean_convergence_episode": float(np.mean(convergence_values)) if convergence_values else None,
            "maximum_convergence_episode": int(max(convergence_values)) if convergence_values else None,
            "mean_wall_minutes_to_convergence": float(np.mean(wall_values) / 60.0) if wall_values else None,
            "mean_final_500_return": float(np.mean([row["final_500_return"] for row in rows])),
            "mean_final_500_arrival": float(np.mean([row["final_500_arrival"] for row in rows])),
        }
        result["configurations"].append({"label": label, "runs": rows, "aggregate": aggregate})
        plot_data[label] = {
            "return": np.vstack(return_curves).mean(axis=0),
            "arrival": np.vstack(arrival_curves).mean(axis=0),
            "wall_minutes": aggregate["mean_wall_minutes_to_convergence"],
        }

    eligible = [
        row for row in result["configurations"]
        if row["aggregate"]["converged_seed_count"] == len(args.seeds)
        and row["aggregate"]["robust_checkpoint_count"] == len(args.seeds)
        and row["aggregate"]["mean_wall_minutes_to_convergence"] is not None
    ]
    winner = min(
        eligible,
        key=lambda row: row["aggregate"]["mean_wall_minutes_to_convergence"],
        default=None,
    )
    result["recommended_configuration"] = winner["label"] if winner else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for label, values in plot_data.items():
        x = np.arange(100, 100 + len(values["return"]))
        axes[0].plot(x, values["return"], label=label)
        axes[1].plot(x, values["arrival"], label=label)
    wall_labels = [label for label, values in plot_data.items() if values["wall_minutes"] is not None]
    wall_values = [plot_data[label]["wall_minutes"] for label in wall_labels]
    axes[2].bar(wall_labels, wall_values)
    axes[0].set(title="4-seed mean return", xlabel="Episode", ylabel="Return")
    axes[1].set(title="4-seed mean arrival time", xlabel="Episode", ylabel="Seconds")
    axes[2].set(title="Mean wall time to convergence", xlabel="Minibatch", ylabel="Minutes")
    for ax in axes[:2]:
        ax.grid(alpha=0.22)
        ax.legend()
    axes[2].grid(axis="y", alpha=0.22)
    fig.suptitle("Minibatch convergence comparison (100-episode moving averages)")
    figure_path = args.output_dir / "minibatch_convergence_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    json_path = args.output_dir / "minibatch_convergence_comparison.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for configuration in result["configurations"]:
        print(configuration["label"], configuration["aggregate"], flush=True)
    print(f"Recommended: {result['recommended_configuration']}", flush=True)
    print(f"Saved: {figure_path.resolve()}", flush=True)
    print(f"Saved: {json_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
