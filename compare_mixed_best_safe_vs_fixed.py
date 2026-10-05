"""Compare a mixed-demand Actor with the best fixed far probability per condition."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from evac_env import EvacuationConfig
from training import MAPPOAgent, PPOConfig, _validation_episode


METRICS = (
    "mean_agent_reward",
    "mean_arrival_time_all_agents",
    "far_fraction",
    "failure_count",
    "capacity_rejection_count",
    "timeout_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--actor", default="best_safe_actor.pt")
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--coarse-step", type=float, default=0.05)
    parser.add_argument("--fine-step", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def summarize(rows: list[dict]) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[metric] = float(values.mean())
        result[f"{metric}_sd"] = float(values.std(ddof=0))
    result["max_failure_count"] = int(max(row["failure_count"] for row in rows))
    return result


def evaluate(
    config: EvacuationConfig,
    agent: MAPPOAgent,
    seeds: list[int],
    condition: tuple[int, int, int],
    fixed_far_probability: float | None,
) -> dict[str, float]:
    return summarize([
        _validation_episode(
            config,
            agent,
            seed,
            fixed_far_probability=fixed_far_probability,
            episode_condition=condition,
        )
        for seed in seeds
    ])


def probability_grid(center: float, step: float, radius: float = 0.05) -> list[float]:
    start = max(0.0, center - radius)
    stop = min(1.0, center + radius)
    count = int(round((stop - start) / step))
    return sorted({round(start + index * step, 10) for index in range(count + 1)})


def make_figure(records: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    panels = (
        ("return", "Mean episode return / agent", "Return", False),
        ("arrival", "Mean arrival time: all agents", "Seconds", False),
        ("far_fraction", "Actual far-choice fraction", "Fraction", False),
        ("failures", "Failed agents", "Agents", True),
    )
    colors = {0.6: "#5e3c99", 0.7: "#1f9e89", 0.8: "#a8db34"}
    for axis, (key, title, ylabel, nonnegative) in zip(axes.flat, panels):
        for ratio in (0.6, 0.7, 0.8):
            subset = sorted(
                [record for record in records if np.isclose(record["near_ratio"], ratio)],
                key=lambda record: record["num_agents"],
            )
            x = [record["num_agents"] for record in subset]
            actor = [record[f"actor_{key}"] for record in subset]
            fixed = [record[f"fixed_{key}"] for record in subset]
            actor_sd = [record[f"actor_{key}_sd"] for record in subset]
            fixed_sd = [record[f"fixed_{key}_sd"] for record in subset]
            axis.errorbar(
                x, actor, yerr=actor_sd, color=colors[ratio], marker="o",
                linewidth=1.8, capsize=3, label=f"Actor, near={ratio:.1f}N",
            )
            axis.errorbar(
                x, fixed, yerr=fixed_sd, color=colors[ratio], marker="s",
                linestyle="--", linewidth=1.5, capsize=3,
                label=f"Best fixed, near={ratio:.1f}N",
            )
        axis.set_title(title)
        axis.set_xlabel("Number of agents")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.set_xticks([1000, 2000, 3000, 4000, 5000])
        if nonnegative:
            axis.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
        ncol=3, frameon=False,
    )
    fig.suptitle(
        "Mixed-demand generalization: best-safe Actor vs best fixed probability",
        y=0.99,
    )
    fig.subplots_adjust(top=0.84, hspace=0.32, wspace=0.24)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    output_dir = (args.output_dir or run / "mixed_best_safe_vs_fixed").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = EvacuationConfig(**json.loads((run / "environment_config.json").read_text()))
    ppo_config = PPOConfig(**json.loads((run / "ppo_config.json").read_text()))
    metadata = json.loads((run / "run_metadata.json").read_text())
    conditions = [tuple(map(int, condition)) for condition in metadata["mixed_demand_conditions"]]
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    agent = MAPPOAgent(ppo_config, torch.device("cpu"))
    try:
        actor_state = torch.load(run / args.actor, map_location="cpu", weights_only=True)
    except TypeError:
        actor_state = torch.load(run / args.actor, map_location="cpu")
    agent.actor.load_state_dict(actor_state)
    agent.actor.eval()

    records = []
    details = []
    coarse_probabilities = np.arange(0.0, 1.0 + args.coarse_step / 2, args.coarse_step)
    for condition_index, condition in enumerate(conditions, start=1):
        demand, near_capacity, far_capacity = condition
        actor = evaluate(config, agent, seeds, condition, None)
        fixed_results: dict[float, dict[str, float]] = {}
        for probability in coarse_probabilities:
            probability = round(float(probability), 10)
            fixed_results[probability] = evaluate(
                config, agent, seeds, condition, probability
            )
        coarse_best = max(
            fixed_results,
            key=lambda probability: (
                fixed_results[probability]["mean_agent_reward"],
                -fixed_results[probability]["mean_arrival_time_all_agents"],
            ),
        )
        for probability in probability_grid(coarse_best, args.fine_step):
            if probability not in fixed_results:
                fixed_results[probability] = evaluate(
                    config, agent, seeds, condition, probability
                )
        best_probability = max(
            fixed_results,
            key=lambda probability: (
                fixed_results[probability]["mean_agent_reward"],
                -fixed_results[probability]["mean_arrival_time_all_agents"],
            ),
        )
        fixed = fixed_results[best_probability]
        near_ratio = near_capacity / demand
        record = {
            "num_agents": demand,
            "near_capacity": near_capacity,
            "far_capacity": far_capacity,
            "near_ratio": near_ratio,
            "best_fixed_probability": best_probability,
            "actor_return": actor["mean_agent_reward"],
            "actor_return_sd": actor["mean_agent_reward_sd"],
            "fixed_return": fixed["mean_agent_reward"],
            "fixed_return_sd": fixed["mean_agent_reward_sd"],
            "return_difference_actor_minus_fixed": (
                actor["mean_agent_reward"] - fixed["mean_agent_reward"]
            ),
            "actor_arrival": actor["mean_arrival_time_all_agents"],
            "actor_arrival_sd": actor["mean_arrival_time_all_agents_sd"],
            "fixed_arrival": fixed["mean_arrival_time_all_agents"],
            "fixed_arrival_sd": fixed["mean_arrival_time_all_agents_sd"],
            "arrival_difference_actor_minus_fixed": (
                actor["mean_arrival_time_all_agents"]
                - fixed["mean_arrival_time_all_agents"]
            ),
            "actor_far_fraction": actor["far_fraction"],
            "actor_far_fraction_sd": actor["far_fraction_sd"],
            "fixed_far_fraction": fixed["far_fraction"],
            "fixed_far_fraction_sd": fixed["far_fraction_sd"],
            "actor_failures": actor["failure_count"],
            "actor_failures_sd": actor["failure_count_sd"],
            "actor_max_failures": actor["max_failure_count"],
            "fixed_failures": fixed["failure_count"],
            "fixed_failures_sd": fixed["failure_count_sd"],
            "fixed_max_failures": fixed["max_failure_count"],
            "actor_rejections": actor["capacity_rejection_count"],
            "fixed_rejections": fixed["capacity_rejection_count"],
            "actor_timeouts": actor["timeout_count"],
            "fixed_timeouts": fixed["timeout_count"],
        }
        records.append(record)
        details.append({
            "condition": list(condition),
            "actor": actor,
            "best_fixed_probability": best_probability,
            "best_fixed": fixed,
            "fixed_sweep": {
                str(probability): result
                for probability, result in sorted(fixed_results.items())
            },
        })
        print(
            f"[{condition_index:02d}/{len(conditions)}] N={demand} "
            f"near={near_capacity} best_p={best_probability:.2f} "
            f"actor_return={actor['mean_agent_reward']:.2f} "
            f"fixed_return={fixed['mean_agent_reward']:.2f} "
            f"actor_fail={actor['failure_count']:.1f} "
            f"fixed_fail={fixed['failure_count']:.1f}",
            flush=True,
        )

    csv_path = output_dir / "best_safe_actor_vs_best_fixed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "best_safe_actor_vs_best_fixed.json").write_text(
        json.dumps({
            "run": str(run),
            "actor": args.actor,
            "evaluation_seeds": seeds,
            "records": records,
            "details": details,
        }, indent=2),
        encoding="utf-8",
    )
    figure_path = output_dir / "best_safe_actor_vs_best_fixed.png"
    make_figure(records, figure_path)
    print(f"Saved {csv_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
