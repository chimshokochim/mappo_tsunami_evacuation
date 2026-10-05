"""Select saved Actors that beat the best fixed policy on return and arrival."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from evac_env import EvacuationConfig
from training import MAPPOAgent, PPOConfig, evaluate_fixed_validation_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[7, 17, 27, 37])
    parser.add_argument(
        "--validation-seeds", type=int, nargs="+",
        default=[10000, 10001, 10002, 10003, 10004],
    )
    parser.add_argument(
        "--far-probabilities", type=float, nargs="+",
        default=[value / 100.0 for value in range(24, 37)],
    )
    parser.add_argument(
        "--summary", type=Path,
        default=Path("joint_checkpoint_selection.json"),
    )
    return parser.parse_args()


def load_dataclass(path: Path, cls):
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in raw.items() if key in allowed})


def find_run(parent: Path, seed: int) -> Path:
    candidates = [
        path.parent for path in parent.rglob("run_metadata.json")
        if json.loads(path.read_text(encoding="utf-8")).get("seed") == seed
    ]
    if not candidates:
        raise FileNotFoundError(f"No run for training seed {seed} below {parent}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_candidate_states(run_dir: Path) -> list[tuple[str, int | None, dict]]:
    candidates: list[tuple[str, int | None, dict]] = []
    actor_path = run_dir / "actor.pt"
    if actor_path.exists():
        candidates.append(("actor.pt", None, torch.load(
            actor_path, map_location="cpu", weights_only=True
        )))
    latest_path = run_dir / "latest_checkpoint.pt"
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        candidates.append((
            "latest_checkpoint.pt",
            int(checkpoint["episode_completed"]),
            checkpoint["actor_state_dict"],
        ))
    if not candidates:
        raise FileNotFoundError(f"No Actor candidates in {run_dir}")
    return candidates


def atomic_save(state_dict: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state_dict, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not args.far_probabilities:
        raise ValueError("--far-probabilities must not be empty")
    if any(not 0.0 <= probability <= 1.0 for probability in args.far_probabilities):
        raise ValueError("--far-probabilities must be within [0, 1]")

    summary = {
        "training_seeds": args.training_seeds,
        "validation_seeds": args.validation_seeds,
        "far_probabilities": args.far_probabilities,
        "runs": [],
    }
    for training_seed in args.training_seeds:
        run_dir = find_run(args.runs, training_seed)
        env_config = load_dataclass(run_dir / "environment_config.json", EvacuationConfig)
        ppo_config = load_dataclass(run_dir / "ppo_config.json", PPOConfig)
        agent = MAPPOAgent(ppo_config, torch.device("cpu"))

        fixed_rows = []
        for probability in args.far_probabilities:
            metrics = evaluate_fixed_validation_seeds(
                env_config, agent, args.validation_seeds,
                fixed_far_probability=probability,
            )
            fixed_rows.append({"far_probability": probability, **metrics})
        baseline = max(
            fixed_rows, key=lambda row: row["mean_episode_return_per_agent"]
        )

        candidate_rows = []
        eligible_candidates = []
        for source, episode, state_dict in load_candidate_states(run_dir):
            agent.actor.load_state_dict(state_dict)
            metrics = evaluate_fixed_validation_seeds(
                env_config, agent, args.validation_seeds
            )
            eligible = (
                metrics["failure_count"] <= baseline["failure_count"]
                and metrics["mean_episode_return_per_agent"]
                > baseline["mean_episode_return_per_agent"]
                and metrics["mean_arrival_time_with_timeouts"]
                < baseline["mean_arrival_time_with_timeouts"]
            )
            row = {
                "source": source,
                "episode": episode,
                **metrics,
                "eligible": eligible,
            }
            candidate_rows.append(row)
            if eligible:
                eligible_candidates.append((row, state_dict))

        selected = None
        if eligible_candidates:
            selected, selected_state = max(
                eligible_candidates,
                key=lambda item: (
                    item[0]["mean_episode_return_per_agent"],
                    -item[0]["mean_arrival_time_with_timeouts"],
                ),
            )
            atomic_save(selected_state, run_dir / "best_joint_actor.pt")

        result = {
            "training_seed": training_seed,
            "run_dir": str(run_dir.resolve()),
            "fixed_baseline": baseline,
            "fixed_sweep": fixed_rows,
            "candidates": candidate_rows,
            "selected": selected,
        }
        summary["runs"].append(result)
        selected_text = (
            f"{selected['source']} R={selected['mean_episode_return_per_agent']:.3f} "
            f"T={selected['mean_arrival_time_with_timeouts']:.2f}s"
            if selected is not None else "none"
        )
        print(
            f"seed={training_seed} fixed: R={baseline['mean_episode_return_per_agent']:.3f} "
            f"T={baseline['mean_arrival_time_with_timeouts']:.2f}s; "
            f"joint checkpoint: {selected_text}",
            flush=True,
        )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {args.summary.resolve()}")


if __name__ == "__main__":
    main()
