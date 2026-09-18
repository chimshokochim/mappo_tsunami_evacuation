"""Evaluate capacity-aware Actors across population sizes and capacity ratios."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from capacity_stress_test import METRICS, discover_runs, sample_sd
from evaluation_utils import aggregate_numeric, load_actor_and_config, run_evaluation_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[7, 17, 27, 37])
    parser.add_argument("--agent-counts", type=int, nargs="+", default=[1000, 2000, 3000, 4000, 5000])
    parser.add_argument("--near-capacity-ratios", type=float, nargs="+", default=[0.6, 0.7, 0.8])
    parser.add_argument("--far-capacity-ratio", type=float, default=0.8)
    parser.add_argument("--reference-agents", type=int, default=3000)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=list(range(400, 420)))
    parser.add_argument(
        "--far-probabilities", type=float, nargs="+",
        default=[value / 10.0 for value in range(11)],
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs_capacity_demand_generalization")
    )
    return parser.parse_args()


def density_samples(rows: list[dict], key: str) -> np.ndarray:
    arrays = [np.asarray(row["trace"][key], dtype=float) for row in rows]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=float)


def distribution_summary(samples: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(samples)),
        "min": float(np.min(samples)),
        "p50": float(np.percentile(samples, 50)),
        "p90": float(np.percentile(samples, 90)),
        "p95": float(np.percentile(samples, 95)),
        "max": float(np.max(samples)),
    }


def mean_across_training_seeds(rows: list[dict]) -> dict[str, float]:
    return {key: float(np.nanmean([row[key] for row in rows])) for key in METRICS}


def make_config(config, agents: int, near_ratio: float, far_ratio: float):
    return replace(
        config,
        num_agents=agents,
        near_shelter_capacity=int(round(near_ratio * agents)),
        near_shelter_capacity_candidates=(),
        far_shelter_capacity=int(round(far_ratio * agents)),
        closure_edge=None,
        random_blockage_probability=0.0,
        shelter_capacity_mode="reject",
        observation_schema="capacity_absolute_and_ratio",
        include_capacity_observation=True,
        include_shelter_capacity_observation=True,
    )


def main() -> None:
    args = parse_args()
    if args.reference_agents not in args.agent_counts:
        raise ValueError("--reference-agents must be included in --agent-counts")
    if any(value <= 0 for value in args.agent_counts):
        raise ValueError("Agent counts must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.near_capacity_ratios):
        raise ValueError("Near-capacity ratios must be in [0, 1]")
    if not 0.0 <= args.far_capacity_ratio <= 1.0:
        raise ValueError("Far-capacity ratio must be in [0, 1]")
    if any(value + args.far_capacity_ratio < 1.0 for value in args.near_capacity_ratios):
        raise ValueError("Every near/far capacity ratio pair must accommodate all agents")
    if any(not 0.0 <= value <= 1.0 for value in args.far_probabilities):
        raise ValueError("Far probabilities must be in [0, 1]")

    run_dirs = discover_runs(args.runs, args.training_seeds)
    loaded = {}
    reference_config = None
    for seed, run_dir in run_dirs.items():
        actor, config, actor_path = load_actor_and_config(run_dir)
        if reference_config is None:
            reference_config = config
        elif config.observation_schema != reference_config.observation_schema:
            raise ValueError("Training runs use incompatible observation schemas")
        loaded[seed] = (actor, config, actor_path)
    assert reference_config is not None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "capacity_demand_progress.json"
    signature = {
        "training_seeds": args.training_seeds,
        "evaluation_seeds": args.evaluation_seeds,
        "agent_counts": args.agent_counts,
        "near_capacity_ratios": args.near_capacity_ratios,
        "far_capacity_ratio": args.far_capacity_ratio,
        "reference_agents": args.reference_agents,
        "far_probabilities": args.far_probabilities,
    }
    result = {
        **signature,
        "capacity_mode": "reject",
        "trained_runs": {str(seed): str(path.resolve()) for seed, path in run_dirs.items()},
        "trained_actor_error_bars": "sample SD across trained-policy seeds after averaging evaluation episodes",
        "fixed_policy_error_bars": "sample SD across evaluation episodes",
        "density_reference_by_near_ratio": {},
        "demand_results": [],
    }
    if progress_path.exists():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        for key, value in signature.items():
            if saved.get(key) != value:
                raise ValueError(
                    f"Existing progress uses a different {key}; choose another --output-dir"
                )
        result = saved
        print(
            f"Resuming {progress_path}: {len(result['demand_results'])} conditions complete",
            flush=True,
        )

    # Establish the N=3000 density range separately for every capacity ratio.
    for near_ratio in args.near_capacity_ratios:
        ratio_key = f"{near_ratio:.6f}"
        if ratio_key in result["density_reference_by_near_ratio"]:
            continue
        config = make_config(
            reference_config, args.reference_agents, near_ratio, args.far_capacity_ratio
        )
        pooled_rows = []
        for training_seed in args.training_seeds:
            actor = loaded[training_seed][0]
            pooled_rows.extend(
                run_evaluation_episode(config, seed, actor=actor, collect_trace=True)
                for seed in args.evaluation_seeds
            )
        near = density_samples(pooled_rows, "decision_raw_density_near")
        far = density_samples(pooled_rows, "decision_raw_density_far")
        result["density_reference_by_near_ratio"][ratio_key] = {
            "near": distribution_summary(near),
            "far": distribution_summary(far),
            "difference": distribution_summary(near - far),
        }
        progress_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved N={args.reference_agents} density reference for near_ratio={near_ratio:.2f}")

    completed = {
        (int(row["num_agents"]), float(row["near_capacity_ratio"]))
        for row in result["demand_results"]
    }
    for agents in args.agent_counts:
        for near_ratio in args.near_capacity_ratios:
            condition = (agents, near_ratio)
            if condition in completed:
                print(f"Skipping completed N={agents}, near_ratio={near_ratio:.2f}", flush=True)
                continue
            config = make_config(reference_config, agents, near_ratio, args.far_capacity_ratio)
            actor_seed_results = []
            pooled_actor_rows = []
            for training_seed in args.training_seeds:
                actor = loaded[training_seed][0]
                episode_rows = [
                    run_evaluation_episode(config, seed, actor=actor, collect_trace=True)
                    for seed in args.evaluation_seeds
                ]
                pooled_actor_rows.extend(episode_rows)
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
                        config, seed, fixed_far_probability=far_probability
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
            actor_sd = sample_sd(actor_seed_results)

            near = density_samples(pooled_actor_rows, "decision_raw_density_near")
            far = density_samples(pooled_actor_rows, "decision_raw_density_far")
            difference = near - far
            observed_near = density_samples(pooled_actor_rows, "decision_density_near")
            observed_far = density_samples(pooled_actor_rows, "decision_density_far")
            ref = result["density_reference_by_near_ratio"][f"{near_ratio:.6f}"]
            density_ood = {}
            for name, values in (("near", near), ("far", far), ("difference", difference)):
                density_ood[name] = {
                    **distribution_summary(values),
                    "percent_above_reference_p95": float(100 * np.mean(values > ref[name]["p95"])),
                    "percent_above_reference_max": float(100 * np.mean(values > ref[name]["max"])),
                    "percent_below_reference_min": float(100 * np.mean(values < ref[name]["min"])),
                }

            result["demand_results"].append({
                "num_agents": agents,
                "near_capacity_ratio": near_ratio,
                "far_capacity_ratio": args.far_capacity_ratio,
                "environment_config": asdict(config),
                "trained_actor_by_seed": actor_seed_results,
                "trained_actor_mean": actor_mean,
                "trained_actor_training_seed_standard_deviation": actor_sd,
                "fixed_probability_sweep": sweep,
                "best_fixed_probability": best_fixed,
                "return_gap_best_fixed_minus_actor": (
                    best_fixed["mean_episode_return_per_agent"]
                    - actor_mean["mean_episode_return_per_agent"]
                ),
                "decision_density_distribution": density_ood,
                "actor_input_saturation": {
                    "near_at_one_percent": float(100 * np.mean(observed_near >= 1.0)),
                    "far_at_one_percent": float(100 * np.mean(observed_far >= 1.0)),
                },
            })
            progress_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(
                f"N={agents} near_ratio={near_ratio:.2f} "
                f"actor_reward={actor_mean['mean_episode_return_per_agent']:.3f} "
                f"arrival={actor_mean['mean_arrival_time_with_timeouts']:.1f}s "
                f"far={actor_mean['sampled_far_fraction']:.3f} "
                f"failures={actor_mean['failure_count']:.2f} "
                f"best_fixed_far={best_fixed['nominal_far_probability']:.1f}",
                flush=True,
            )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    panels = (
        ("mean_episode_return_per_agent", "Mean episode return / agent", None),
        ("mean_arrival_time_with_timeouts", "Mean arrival time: all agents", "Seconds"),
        ("sampled_far_fraction", "Actual far-choice fraction", "Fraction"),
        ("failure_count", "Failed agents", "Agents"),
        ("capacity_rejection_count", "Capacity rejections", "Agents"),
    )
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(args.near_capacity_ratios)))
    for color, near_ratio in zip(colors, args.near_capacity_ratios):
        rows = sorted(
            (row for row in result["demand_results"] if row["near_capacity_ratio"] == near_ratio),
            key=lambda row: row["num_agents"],
        )
        counts = np.asarray([row["num_agents"] for row in rows])
        for axis, (key, title, ylabel) in zip(axes.flat[:5], panels):
            means = [row["trained_actor_mean"][key] for row in rows]
            sds = [row["trained_actor_training_seed_standard_deviation"][key] for row in rows]
            fixed_key = key
            fixed_values = [row["best_fixed_probability"][fixed_key] for row in rows]
            axis.errorbar(
                counts, means, yerr=sds, fmt="o-", color=color, capsize=3,
                label=f"Actor near cap={near_ratio:.1f}N",
            )
            axis.plot(
                counts, fixed_values, "s--", color=color, alpha=0.65,
                label=f"best fixed {near_ratio:.1f}N",
            )
            axis.set_title(title)
            axis.set_xlabel("Number of agents")
            if ylabel:
                axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
        ood_axis = axes.flat[5]
        ood_axis.plot(
            counts,
            [row["decision_density_distribution"]["near"]["percent_above_reference_max"] for row in rows],
            "o-", color=color, label=f"> N=3000 max, near cap={near_ratio:.1f}N",
        )
        ood_axis.plot(
            counts,
            [row["actor_input_saturation"]["near_at_one_percent"] for row in rows],
            "^:", color=color, alpha=0.8, label=f"input clipped, {near_ratio:.1f}N",
        )

    axes.flat[2].set_ylim(0.0, 1.0)
    axes.flat[3].set_ylim(bottom=0.0)
    axes.flat[4].set_ylim(bottom=0.0)
    axes.flat[5].set_title("Near-density distribution shift")
    axes.flat[5].set_xlabel("Number of agents")
    axes.flat[5].set_ylabel("Decisions (%)")
    axes.flat[5].grid(alpha=0.25)
    for axis in axes.flat:
        axis.legend(fontsize=7)
    figure.suptitle(
        f"Demand generalization: {len(args.training_seeds)} trained seeds, "
        f"{len(args.evaluation_seeds)} evaluation seeds per condition"
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    json_path = args.output_dir / f"capacity_demand_generalization_{stamp}.json"
    png_path = args.output_dir / f"capacity_demand_generalization_{stamp}.png"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    print(f"Saved: {png_path.resolve()}")
    print(f"Saved: {json_path.resolve()}")


if __name__ == "__main__":
    main()
