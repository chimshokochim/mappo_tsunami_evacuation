"""Run minibatch 1024/2048 experiments and compare convergence overnight."""

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
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37])
    parser.add_argument("--minibatch-sizes", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--episodes", type=int, default=6000)
    parser.add_argument(
        "--baseline-runs", type=Path,
        default=Path("outputs_capacity_fixed2200_teamgamma1_validated"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs_minibatch_ablation"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def matching_run(parent: Path, seed: int, minibatch_size: int) -> Path | None:
    matches = []
    for checkpoint_path in parent.rglob("latest_checkpoint.pt"):
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        diagnostics = checkpoint.get("diagnostics", {})
        config = diagnostics.get("ppo_config", {})
        if (
            int(diagnostics.get("seed", -1)) == seed
            and int(config.get("minibatch_size", -1)) == minibatch_size
        ):
            matches.append(checkpoint_path.parent)
    if not matches:
        return None
    return max(matches, key=lambda path: (path / "latest_checkpoint.pt").stat().st_mtime)


def checkpoint_episode(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint["episode_completed"])


def training_command(
    training_script: Path, seed: int, minibatch_size: int, episodes: int,
    output_dir: Path, resume: Path | None,
) -> list[str]:
    command = [
        sys.executable, str(training_script),
        "--episodes", str(episodes),
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
        "--minibatch-size", str(minibatch_size),
        "--profile-runtime",
        "--log-every-updates", "25",
        "--validation-every-updates", "50",
        "--checkpoint-every-updates", "50",
        "--validation-start-episode", "2500",
        "--validation-seeds", "10000", "10001", "10002", "10003", "10004",
        "--robust-joint-min-return-margin", "0.5",
        "--robust-joint-min-arrival-margin", "1.0",
        "--output-dir", str(output_dir),
    ]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return command


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")
    if len(set(args.minibatch_sizes)) != len(args.minibatch_sizes):
        raise ValueError("--minibatch-sizes contains duplicates")
    if any(size <= 0 for size in args.minibatch_sizes):
        raise ValueError("minibatch sizes must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_script = Path(__file__).with_name("training.py")
    analyzer = Path(__file__).with_name("analyze_minibatch_ablation.py")
    manifest_path = args.output_dir / "overnight_manifest.json"
    manifest = {
        "started_at": datetime.now().isoformat(),
        "seeds": args.seeds,
        "minibatch_sizes": args.minibatch_sizes,
        "episodes": args.episodes,
        "baseline_runs": str(args.baseline_runs.resolve()),
        "completed": [],
    }

    for minibatch_size in args.minibatch_sizes:
        config_dir = args.output_dir / f"minibatch_{minibatch_size}"
        config_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            run_dir = matching_run(config_dir, seed, minibatch_size)
            resume = None
            if run_dir is not None:
                checkpoint = run_dir / "latest_checkpoint.pt"
                completed = checkpoint_episode(checkpoint)
                if completed >= args.episodes:
                    print(
                        f"Skipping minibatch={minibatch_size}, seed={seed}: "
                        f"already at episode {completed}", flush=True,
                    )
                    manifest["completed"].append({
                        "minibatch_size": minibatch_size, "seed": seed,
                        "run": str(run_dir.resolve()),
                    })
                    continue
                resume = checkpoint
            command = training_command(
                training_script, seed, minibatch_size, args.episodes,
                config_dir, resume,
            )
            print("\nRunning:", " ".join(command), flush=True)
            if args.dry_run:
                continue
            subprocess.run(command, check=True)
            completed_run = matching_run(config_dir, seed, minibatch_size)
            manifest["completed"].append({
                "minibatch_size": minibatch_size,
                "seed": seed,
                "run": str(completed_run.resolve()),
            })
            manifest["last_completed_at"] = datetime.now().isoformat()
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if not args.dry_run:
            subprocess.run([
                sys.executable,
                str(Path(__file__).with_name("plot_all_seed_convergence.py")),
                "--runs", str(config_dir),
                "--seeds", *[str(seed) for seed in args.seeds],
                "--rolling-window", "100",
                "--summary-window", "500",
            ], check=True)

    config_arguments = [f"512={args.baseline_runs}"] + [
        f"{size}={args.output_dir / f'minibatch_{size}'}"
        for size in args.minibatch_sizes
    ]
    analyze_command = [
        sys.executable, str(analyzer),
        "--seeds", *[str(seed) for seed in args.seeds],
        "--output-dir", str(args.output_dir / "comparison"),
    ]
    for value in config_arguments:
        analyze_command.extend(["--config", value])
    print("\nAnalyzing:", " ".join(analyze_command), flush=True)
    if args.dry_run:
        print("Dry run complete; no training was started.")
        return
    subprocess.run(analyze_command, check=True)
    manifest["finished_at"] = datetime.now().isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Overnight experiment complete: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
