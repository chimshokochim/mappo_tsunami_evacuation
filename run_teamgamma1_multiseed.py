"""Run the fixed-capacity, team-return-gamma=1.0 experiment for several seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37])
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--validation-every-updates", type=int, default=100)
    parser.add_argument("--validation-start-episode", type=int, default=2500)
    parser.add_argument(
        "--validation-seeds", type=int, nargs="+",
        default=[10000, 10001, 10002, 10003, 10004],
    )
    parser.add_argument(
        "--joint-fixed-far-probabilities", type=float, nargs="+",
        default=[value / 100.0 for value in range(24, 37)],
        help="fixed policies used to define the joint return/arrival baseline",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_capacity_fixed2200_teamgamma1_multiseed"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.validation_every_updates <= 0:
        raise ValueError("--validation-every-updates must be positive")
    if args.validation_start_episode < 0:
        raise ValueError("--validation-start-episode must be non-negative")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")
    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        raise ValueError("--validation-seeds contains duplicates")

    training_script = Path(__file__).with_name("training.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": datetime.now().isoformat(),
        "seeds": args.seeds,
        "episodes": args.episodes,
        "team_return_gamma": 1.0,
        "near_capacity": 2200,
        "far_capacity": 2400,
        "validation_every_updates": args.validation_every_updates,
        "validation_start_episode": args.validation_start_episode,
        "validation_seeds": args.validation_seeds,
        "joint_fixed_far_probabilities": args.joint_fixed_far_probabilities,
        "completed_seeds": [],
    }
    manifest_path = args.output_dir / "multiseed_manifest.json"

    for seed in args.seeds:
        command = [
            sys.executable,
            str(training_script),
            "--episodes", str(args.episodes),
            "--seed", str(seed),
            "--team-return-gamma", "1.0",
            "--entropy-mode", "annealed",
            "--entropy-start", "0.05",
            "--entropy-end", "0.01",
            "--entropy-anneal-episodes", "3000",
            "--near-capacities", "2200",
            "--far-capacity", "2400",
            "--capacity-mode", "reject",
            "--capacity-observation", "absolute-and-demand-ratio",
            "--profile-runtime",
            "--log-every-updates", "10",
            "--validation-every-updates", str(args.validation_every_updates),
            "--validation-start-episode", str(args.validation_start_episode),
            "--validation-seeds", *[str(value) for value in args.validation_seeds],
            "--joint-fixed-far-probabilities",
            *[str(value) for value in args.joint_fixed_far_probabilities],
            "--output-dir", str(args.output_dir),
        ]
        print("\nRunning:", " ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, check=True)
        manifest["completed_seeds"].append(seed)
        manifest["last_completed_at"] = datetime.now().isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print("Dry run complete; no training was started.")
    else:
        print(f"All requested seeds completed. Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
