"""Stress-test multiple trained capacity-aware Actors in reject mode."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from evaluation_utils import (
    aggregate_numeric,
    load_actor_and_config,
    run_evaluation_episode,
)


DEFAULT_CAPACITIES = (
    2400, 2350, 2300, 2250, 2200, 2150, 2100, 2050,
    2000, 1950, 1900, 1850, 1800, 1700, 1600, 1500,
)
DEFAULT_FAR_PROBABILITIES = tuple(value / 10.0 for value in range(11))
METRICS = (
    "mean_episode_return_per_agent",
    "arrived_only_mean_arrival_time",
    "mean_arrival_time_with_timeouts",
    "arrived_count",
    "failure_count",
    "capacity_rejection_count",
    "timeout_count",
    "sampled_far_fraction",
    "mean_actor_probability_far",
    "mean_effective_probability_far",
    "near_reserved_count",
    "far_reserved_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path, nargs="+", required=True,
        help="run directories or parent directories containing diagnostics.pkl",
    )
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[7, 17, 27, 37])
    parser.add_argument("--near-capacities", type=int, nargs="+", default=list(DEFAULT_CAPACITIES))
    parser.add_argument("--far-capacity", type=int, default=2400)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=list(range(300, 320)))
    parser.add_argument(
        "--far-probabilities", type=float, nargs="+",
        default=list(DEFAULT_FAR_PROBABILITIES),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("capacity_stress_test"))
    return parser.parse_args()


def sample_sd(rows: list[dict], keys: tuple[str, ...] = METRICS) -> dict[str, float]:
    return {
        key: float(np.nanstd([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0
        for key in keys
    }


def discover_runs(paths: list[Path], requested_seeds: list[int]) -> dict[int, Path]:
    candidates: list[Path] = []
    for path in paths:
        if path.name == "diagnostics.pkl" and path.is_file():
            candidates.append(path)
        elif (path / "diagnostics.pkl").is_file():
            candidates.append(path / "diagnostics.pkl")
        elif path.exists():
            candidates.extend(path.rglob("diagnostics.pkl"))
    selected: dict[int, Path] = {}
    for diagnostics_path in candidates:
        with diagnostics_path.open("rb") as handle:
            diagnostics = pickle.load(handle)
        seed = int(diagnostics.get("seed", -1))
        env = diagnostics.get("environment_config", {})
        if seed not in requested_seeds:
            continue
        if env.get("shelter_capacity_mode") != "reject":
            continue
        if env.get("observation_schema") != "capacity_absolute_and_ratio":
            continue
        previous = selected.get(seed)
        if previous is None or diagnostics_path.stat().st_mtime > previous.stat().st_mtime:
            selected[seed] = diagnostics_path
    missing = [seed for seed in requested_seeds if seed not in selected]
    if missing:
        raise FileNotFoundError(f"No completed matching run found for training seeds {missing}")
    return {seed: selected[seed].parent for seed in requested_seeds}


def mean_across_training_seeds(rows: list[dict]) -> dict[str, float]:
    return {key: float(np.nanmean([row[key] for row in rows])) for key in METRICS}


def main() -> None:
    args = parse_args()
    if len(set(args.training_seeds)) != len(args.training_seeds):
        raise ValueError("--training-seeds contains duplicates")
    if len(set(args.evaluation_seeds)) != len(args.evaluation_seeds):
        raise ValueError("--evaluation-seeds contains duplicates")
    if any(value < 0 for value in args.near_capacities) or args.far_capacity < 0:
        raise ValueError("Shelter capacities must be non-negative")
    if any(not 0.0 <= value <= 1.0 for value in args.far_probabilities):
        raise ValueError("Far probabilities must be in [0, 1]")

    run_dirs = discover_runs(args.runs, args.training_seeds)
    loaded = {}
    reference_config = None
    for seed, run_dir in run_dirs.items():
        actor, config, actor_path = load_actor_and_config(run_dir)
        if reference_config is None:
            reference_config = config
        elif (
            config.num_agents != reference_config.num_agents
            or config.observation_schema != reference_config.observation_schema
        ):
            raise ValueError("Training runs use incompatible environment configurations")
        loaded[seed] = (actor, config, actor_path)
    assert reference_config is not None
    if any(capacity + args.far_capacity < reference_config.num_agents for capacity in args.near_capacities):
        raise ValueError("Every capacity pair must be able to accommodate all agents")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "capacity_stress_progress.json"
    run_signature = {
        "training_seeds": args.training_seeds,
        "evaluation_seeds": args.evaluation_seeds,
        "near_capacities": args.near_capacities,
        "far_capacity": args.far_capacity,
        "far_probabilities": args.far_probabilities,
    }
    result = {
        **run_signature,
        "capacity_mode": "reject",
        "trained_runs": {str(seed): str(run_dirs[seed].resolve()) for seed in args.training_seeds},
        "trained_actor_error_bars": "sample SD across trained-policy seeds after averaging evaluation episodes",
        "fixed_policy_error_bars": "sample SD across evaluation episodes",
        "capacity_results": [],
    }
    if progress_path.exists():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        for key, value in run_signature.items():
            if saved.get(key) != value:
                raise ValueError(
                    f"Existing progress uses a different {key}; choose another --output-dir"
                )
        result = saved
        print(
            f"Resuming {progress_path}: "
            f"{len(result['capacity_results'])} capacities already complete",
            flush=True,
        )

    completed_capacities = {row["near_capacity"] for row in result["capacity_results"]}
    for near_capacity in args.near_capacities:
        if near_capacity in completed_capacities:
            print(f"Skipping completed near_capacity={near_capacity}", flush=True)
            continue
        base_config = replace(
            reference_config,
            near_shelter_capacity=near_capacity,
            near_shelter_capacity_candidates=(),
            far_shelter_capacity=args.far_capacity,
            closure_edge=None,
            random_blockage_probability=0.0,
            shelter_capacity_mode="reject",
            observation_schema="capacity_absolute_and_ratio",
            include_capacity_observation=True,
            include_shelter_capacity_observation=True,
        )
        actor_seed_results = []
        for training_seed in args.training_seeds:
            actor = loaded[training_seed][0]
            episode_rows = [
                run_evaluation_episode(base_config, seed, actor=actor)
                for seed in args.evaluation_seeds
            ]
            actor_seed_results.append({
                "training_seed": training_seed,
                "actor_path": str(loaded[training_seed][2].resolve()),
                **aggregate_numeric(episode_rows, METRICS),
                "evaluation_seed_standard_deviation": sample_sd(episode_rows),
            })

        sweep = []
        for far_probability in args.far_probabilities:
            rows = [
                run_evaluation_episode(
                    base_config, seed, fixed_far_probability=far_probability
                )
                for seed in args.evaluation_seeds
            ]
            sweep.append({
                "nominal_far_probability": far_probability,
                **aggregate_numeric(rows, METRICS),
                "evaluation_seed_standard_deviation": sample_sd(rows),
            })
        best_fixed = max(sweep, key=lambda row: row["mean_episode_return_per_agent"])
        actor_mean = mean_across_training_seeds(actor_seed_results)
        actor_training_seed_sd = sample_sd(actor_seed_results)
        result["capacity_results"].append({
            "near_capacity": near_capacity,
            "environment_config": asdict(base_config),
            "trained_actor_by_seed": actor_seed_results,
            "trained_actor_mean": actor_mean,
            "trained_actor_training_seed_standard_deviation": actor_training_seed_sd,
            "fixed_probability_sweep": sweep,
            "best_fixed_probability": best_fixed,
            "return_gap_best_fixed_minus_actor": (
                best_fixed["mean_episode_return_per_agent"]
                - actor_mean["mean_episode_return_per_agent"]
            ),
        })
        progress_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(
            f"near_capacity={near_capacity} "
            f"actor_reward={actor_mean['mean_episode_return_per_agent']:.3f} "
            f"actor_far={actor_mean['sampled_far_fraction']:.3f} "
            f"actor_prob_far={actor_mean['mean_actor_probability_far']:.3f} "
            f"failures={actor_mean['failure_count']:.2f} "
            f"rejections={actor_mean['capacity_rejection_count']:.2f} "
            f"best_fixed_far={best_fixed['nominal_far_probability']:.1f} "
            f"best_fixed_minus_actor="
            f"{result['capacity_results'][-1]['return_gap_best_fixed_minus_actor']:.3f}",
            flush=True,
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    capacities = np.asarray(args.near_capacities)
    means = [row["trained_actor_mean"] for row in result["capacity_results"]]
    sds = [
        row["trained_actor_training_seed_standard_deviation"]
        for row in result["capacity_results"]
    ]
    fixed = [row["best_fixed_probability"] for row in result["capacity_results"]]
    panels = (
        ("mean_episode_return_per_agent", "Mean episode return / agent", None),
        ("mean_arrival_time_with_timeouts", "Mean arrival time: all agents", "Seconds"),
        ("sampled_far_fraction", "Actual far-choice fraction", "Fraction"),
        ("mean_actor_probability_far", "Mean Actor P(far)", "Probability"),
        ("failure_count", "Failed agents", "Agents"),
        ("capacity_rejection_count", "Capacity rejections", "Agents"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for axis, (key, title, ylabel) in zip(axes.flat, panels):
        axis.errorbar(
            capacities, [row[key] for row in means], yerr=[row[key] for row in sds],
            fmt="o-", capsize=4, label="trained Actors (mean +/- SD across training seeds)",
        )
        fixed_key = "mean_effective_probability_far" if key == "mean_actor_probability_far" else key
        axis.plot(
            capacities, [row[fixed_key] for row in fixed], "s--",
            label="best fixed-probability sweep",
        )
        axis.set_title(title)
        axis.set_xlabel("Near shelter capacity")
        if ylabel:
            axis.set_ylabel(ylabel)
        if key in ("sampled_far_fraction", "mean_actor_probability_far"):
            axis.set_ylim(0.0, 1.0)
        if key in ("failure_count", "capacity_rejection_count"):
            axis.set_ylim(bottom=0.0)
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    figure.suptitle(
        f"Capacity stress test: {len(args.training_seeds)} trained seeds, "
        f"{len(args.evaluation_seeds)} evaluation seeds per condition"
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    json_path = args.output_dir / f"capacity_stress_test_{stamp}.json"
    png_path = args.output_dir / f"capacity_stress_test_{stamp}.png"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    print(f"Saved: {png_path.resolve()}")
    print(f"Saved: {json_path.resolve()}")


if __name__ == "__main__":
    main()
