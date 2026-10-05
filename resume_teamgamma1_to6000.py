"""Resume the remaining fixed-capacity training seeds to a common episode budget."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 37])
    parser.add_argument("--episodes", type=int, default=6000)
    parser.add_argument("--validation-every-updates", type=int, default=50)
    parser.add_argument("--min-return-margin", type=float, default=0.5)
    parser.add_argument("--min-arrival-margin", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def find_run(parent: Path, seed: int) -> Path:
    candidates = []
    for metadata_path in parent.rglob("run_metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("seed", -1)) == seed:
            candidates.append(metadata_path.parent)
    if not candidates:
        raise FileNotFoundError(f"No run for seed {seed} below {parent}")
    return max(candidates, key=lambda path: (path / "latest_checkpoint.pt").stat().st_mtime)


def completed_episode(checkpoint_path: Path) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return int(checkpoint["episode_completed"])


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.validation_every_updates <= 0:
        raise ValueError("--validation-every-updates must be positive")
    if args.min_return_margin < 0.0 or args.min_arrival_margin < 0.0:
        raise ValueError("robust margins must be non-negative")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")

    training_script = Path(__file__).with_name("training.py")
    manifest = {
        "started_at": datetime.now().isoformat(),
        "seeds": args.seeds,
        "target_episodes": args.episodes,
        "validation_every_updates": args.validation_every_updates,
        "robust_joint_min_return_margin": args.min_return_margin,
        "robust_joint_min_arrival_margin": args.min_arrival_margin,
        "completed_seeds": [],
    }
    manifest_path = args.runs / "resume_to_6000_manifest.json"

    for seed in args.seeds:
        run_dir = find_run(args.runs, seed)
        checkpoint_path = run_dir / "latest_checkpoint.pt"
        completed = completed_episode(checkpoint_path)
        if completed >= args.episodes:
            print(
                f"Skipping seed {seed}: checkpoint already contains episode {completed}",
                flush=True,
            )
            manifest["completed_seeds"].append(seed)
            continue
        command = [
            sys.executable, str(training_script),
            "--episodes", str(args.episodes),
            "--seed", str(seed),
            "--resume", str(checkpoint_path),
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
            "--validation-start-episode", "2500",
            "--validation-seeds", "10000", "10001", "10002", "10003", "10004",
            "--robust-joint-min-return-margin", str(args.min_return_margin),
            "--robust-joint-min-arrival-margin", str(args.min_arrival_margin),
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
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"All requested seeds completed. Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
