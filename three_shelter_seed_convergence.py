"""Train and compare independent seeds for the capacity-free three-shelter model."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", action="store_true",
        help="train missing seeds and resume matching interrupted runs",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37])
    parser.add_argument("--episodes", type=int, default=12000)
    parser.add_argument("--entropy-mode", choices=["constant", "annealed"], default="annealed")
    parser.add_argument("--num-agents", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument(
        "--road-lengths", type=float, nargs=3, default=[150.0, 225.0, 300.0],
        metavar=("NEAR", "MIDDLE", "FAR"),
    )
    parser.add_argument("--road-width", type=float, default=5.0)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint-every-updates", type=int, default=25)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs_three_shelter_seed_sweep")
    )
    parser.add_argument(
        "--include", type=Path, nargs="*", default=[],
        help="additional completed-run directories to reuse (for example seed 7)",
    )
    parser.add_argument("--rolling-window", type=int, default=100)
    return parser.parse_args()


def rolling_mean(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    width = min(max(1, window), len(values))
    kernel = np.ones(width, dtype=float) / width
    return np.arange(width, len(values) + 1), np.convolve(values, kernel, mode="valid")


def _matching_configuration(
    environment: dict, ppo: dict, args: argparse.Namespace,
) -> bool:
    road_lengths = environment.get("road_lengths", [])
    return (
        int(environment.get("num_agents", -1)) == args.num_agents
        and int(environment.get("max_steps", -1)) == args.max_steps
        and len(road_lengths) == 3
        and np.allclose(road_lengths, args.road_lengths)
        and float(environment.get("road_width", float("nan"))) == args.road_width
        and ppo.get("entropy_mode") == args.entropy_mode
        and int(ppo.get("minibatch_size", -1)) == args.minibatch_size
    )


def _files_under(paths: list[Path], filename: str) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.name == filename:
            files.append(path)
        elif path.exists():
            files.extend(path.rglob(filename))
    return files


def load_completed_runs(
    paths: list[Path], args: argparse.Namespace,
) -> dict[int, tuple[Path, dict]]:
    selected: dict[int, tuple[Path, dict]] = {}
    for diagnostics_path in _files_under(paths, "diagnostics.pkl"):
        try:
            with diagnostics_path.open("rb") as handle:
                diagnostics = pickle.load(handle)
        except (OSError, EOFError, pickle.UnpicklingError):
            continue
        if diagnostics.get("shelters") != ["near", "middle", "far"]:
            continue
        if not _matching_configuration(
            diagnostics.get("environment_config", {}),
            diagnostics.get("ppo_config", {}), args,
        ):
            continue
        if len(diagnostics.get("episode", [])) < args.episodes:
            continue
        seed = int(diagnostics["seed"])
        previous = selected.get(seed)
        if previous is None or diagnostics_path.stat().st_mtime > previous[0].stat().st_mtime:
            selected[seed] = (diagnostics_path, diagnostics)
    return selected


def load_resumable_runs(
    paths: list[Path], args: argparse.Namespace,
) -> dict[int, Path]:
    try:
        import torch
    except ImportError:
        return {}

    selected: dict[int, Path] = {}
    for checkpoint_path in _files_under(paths, "checkpoint_latest.pt"):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError):
            continue
        if not _matching_configuration(
            checkpoint.get("environment_config", {}),
            checkpoint.get("ppo_config", {}), args,
        ):
            continue
        seed = int(checkpoint["seed"])
        previous = selected.get(seed)
        if previous is None or checkpoint_path.stat().st_mtime > previous.stat().st_mtime:
            selected[seed] = checkpoint_path
    return selected


def train_missing_runs(args: argparse.Namespace, search_paths: list[Path]) -> None:
    trainer = Path(__file__).with_name("three_shelter_training.py")
    completed = load_completed_runs(search_paths, args)
    resumable = load_resumable_runs(search_paths, args)
    for seed in args.seeds:
        if seed in completed:
            print(f"Skipping seed {seed}: completed run found at {completed[seed][0].parent}")
            continue
        if seed in resumable:
            run_dir = resumable[seed].parent
            command = [
                sys.executable, str(trainer), "--resume", str(run_dir),
                "--episodes", str(args.episodes), "--device", args.device,
                "--checkpoint-every-updates", str(args.checkpoint_every_updates),
            ]
            print(f"\nResuming seed {seed}: {' '.join(command)}", flush=True)
        else:
            command = [
                sys.executable, str(trainer),
                "--episodes", str(args.episodes),
                "--entropy-mode", args.entropy_mode,
                "--seed", str(seed),
                "--num-agents", str(args.num_agents),
                "--max-steps", str(args.max_steps),
                "--road-lengths", *[str(value) for value in args.road_lengths],
                "--road-width", str(args.road_width),
                "--minibatch-size", str(args.minibatch_size),
                "--device", args.device,
                "--output-dir", str(args.output_dir),
                "--checkpoint-every-updates", str(args.checkpoint_every_updates),
                "--log-every-updates", "25",
            ]
            print(f"\nTraining seed {seed}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
        completed = load_completed_runs(search_paths, args)
        resumable = load_resumable_runs(search_paths, args)


def tail_mean(values: np.ndarray, width: int) -> float:
    return float(np.mean(values[-min(width, len(values)) :]))


def sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    search_paths = [args.output_dir, *args.include]
    if args.train:
        train_missing_runs(args, search_paths)

    runs = load_completed_runs(search_paths, args)
    missing = [seed for seed in args.seeds if seed not in runs]
    if missing:
        raise FileNotFoundError(
            f"No matching {args.episodes}-episode run found for seeds {missing}. "
            "Use --train, or provide completed runs with --include."
        )
    runs = {seed: runs[seed] for seed in args.seeds}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    route_keys = ("sampled_near_fraction", "sampled_middle_fraction", "sampled_far_fraction")
    route_names = ("Near choice fraction", "Middle choice fraction", "Far choice fraction")
    summary: list[dict] = []
    for seed, (path, diagnostics) in runs.items():
        episodes = np.asarray(diagnostics["episode"], dtype=int)
        update_episodes = np.asarray(diagnostics["update_episode"], dtype=int)
        metrics = diagnostics["episode_metrics"]
        updates = diagnostics["update_metrics"]
        returns = np.asarray(metrics["mean_episode_return_per_agent"], dtype=float)
        arrivals = np.asarray(metrics["mean_arrival_time_with_timeouts"], dtype=float)

        x, smoothed = rolling_mean(returns, args.rolling_window)
        axes[0, 0].plot(episodes[x - 1], smoothed, label=f"seed {seed}")
        x, smoothed = rolling_mean(arrivals, args.rolling_window)
        axes[0, 1].plot(episodes[x - 1], smoothed, label=f"seed {seed}")
        route_means = {}
        for ax, key, name in zip((axes[0, 2], axes[1, 0], axes[1, 1]), route_keys, route_names):
            values = np.asarray(metrics[key], dtype=float)
            x, smoothed = rolling_mean(values, args.rolling_window)
            ax.plot(episodes[x - 1], smoothed, label=f"seed {seed}")
            route_means[key] = tail_mean(values, args.rolling_window)

        entropy = np.asarray(updates["policy_entropy"], dtype=float)
        update_window = max(1, args.rolling_window // 4)
        x, smoothed = rolling_mean(entropy, update_window)
        axes[1, 2].plot(update_episodes[x - 1], smoothed, label=f"seed {seed}")
        summary.append({
            "seed": seed,
            "diagnostics": str(path.resolve()),
            "episodes": len(episodes),
            "final_mean_return": tail_mean(returns, args.rolling_window),
            "final_mean_arrival_all": tail_mean(arrivals, args.rolling_window),
            "final_sampled_near_fraction": route_means["sampled_near_fraction"],
            "final_sampled_middle_fraction": route_means["sampled_middle_fraction"],
            "final_sampled_far_fraction": route_means["sampled_far_fraction"],
            "final_policy_entropy": tail_mean(entropy, update_window),
            "final_actor_loss": tail_mean(np.asarray(updates["actor_loss"], dtype=float), update_window),
            "final_critic_loss": tail_mean(np.asarray(updates["critic_loss"], dtype=float), update_window),
            "final_approx_kl": tail_mean(np.asarray(updates["approx_kl"], dtype=float), update_window),
            "final_clip_fraction": tail_mean(np.asarray(updates["clip_fraction"], dtype=float), update_window),
            "final_failed_agents": tail_mean(
                np.asarray(metrics["failure_count"], dtype=float), args.rolling_window
            ),
        })

    titles = [
        "Mean episode return / agent", "Mean arrival time (all agents)",
        *route_names, "Categorical policy entropy",
    ]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0, 0].set_ylabel("Return")
    axes[0, 1].set_ylabel("Seconds")
    for ax in (axes[0, 2], axes[1, 0], axes[1, 1]):
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction")
    axes[1, 2].set_ylabel("Nats")
    axes[1, 2].axhline(np.log(3), color="black", linestyle="--", linewidth=0.9, label="ln(3)")
    axes[1, 2].legend()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    png_path = args.output_dir / f"three_shelter_seed_convergence_{stamp}.png"
    json_path = args.output_dir / f"three_shelter_seed_convergence_{stamp}.json"
    figure.savefig(png_path, dpi=160)
    plt.close(figure)

    aggregate: dict[str, float | int] = {"seed_count": len(summary)}
    aggregate_metrics = (
        "final_mean_return", "final_mean_arrival_all", "final_sampled_near_fraction",
        "final_sampled_middle_fraction", "final_sampled_far_fraction",
        "final_policy_entropy", "final_actor_loss", "final_critic_loss",
        "final_approx_kl", "final_clip_fraction", "final_failed_agents",
    )
    for metric in aggregate_metrics:
        values = [float(row[metric]) for row in summary]
        aggregate[f"{metric}_mean_across_seeds"] = float(np.mean(values))
        aggregate[f"{metric}_sample_std_across_seeds"] = sample_std(values)
    json_path.write_text(
        json.dumps({"runs": summary, "aggregate": aggregate}, indent=2), encoding="utf-8"
    )

    for row in summary:
        print(
            f"seed={row['seed']} return={row['final_mean_return']:.3f} "
            f"arrival={row['final_mean_arrival_all']:.1f}s "
            f"choices=({row['final_sampled_near_fraction']:.3f},"
            f"{row['final_sampled_middle_fraction']:.3f},"
            f"{row['final_sampled_far_fraction']:.3f}) "
            f"entropy={row['final_policy_entropy']:.4f} "
            f"critic_loss={row['final_critic_loss']:.6f}"
        )
    print(
        "Across seeds: "
        f"return={aggregate['final_mean_return_mean_across_seeds']:.3f} +/- "
        f"{aggregate['final_mean_return_sample_std_across_seeds']:.3f}, "
        f"arrival={aggregate['final_mean_arrival_all_mean_across_seeds']:.1f} +/- "
        f"{aggregate['final_mean_arrival_all_sample_std_across_seeds']:.1f}s"
    )
    print(f"Saved: {png_path.resolve()}")
    print(f"Saved: {json_path.resolve()}")


if __name__ == "__main__":
    main()
