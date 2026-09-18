"""Run and compare independent capacity-constrained MAPPO training seeds."""

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
        help="train only requested seeds that are not already available",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37])
    parser.add_argument("--episodes", type=int, default=12000)
    parser.add_argument("--entropy-mode", choices=["constant", "annealed"], default="annealed")
    parser.add_argument("--num-agents", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument(
        "--near-capacities", type=int, nargs="+",
        default=[2400, 2300, 2200, 2100, 2000, 1900, 1800],
    )
    parser.add_argument("--far-capacity", type=int, default=2400)
    parser.add_argument("--failure-penalty", type=float, default=-500.0)
    parser.add_argument("--capacity-mode", choices=["reject"], default="reject")
    parser.add_argument(
        "--capacity-observation",
        choices=["absolute-and-demand-ratio"],
        default="absolute-and-demand-ratio",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs_capacity_reject_seed_sweep")
    )
    parser.add_argument("--include", type=Path, nargs="*", default=[])
    parser.add_argument("--rolling-window", type=int, default=100)
    return parser.parse_args()


def rolling_mean(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    width = min(window, len(values))
    kernel = np.ones(width) / width
    return np.arange(width, len(values) + 1), np.convolve(values, kernel, mode="valid")


def load_runs(
    paths: list[Path], entropy_mode: str, near_capacities: list[int], far_capacity: int,
    failure_penalty: float, capacity_mode: str, capacity_observation: str,
) -> dict[int, tuple[Path, dict]]:
    candidates = []
    for path in paths:
        if path.is_file() and path.name == "diagnostics.pkl":
            candidates.append(path)
        elif path.exists():
            candidates.extend(path.rglob("diagnostics.pkl"))
    selected = {}
    for diagnostics_path in candidates:
        with diagnostics_path.open("rb") as handle:
            diagnostics = pickle.load(handle)
        if diagnostics.get("entropy_mode") != entropy_mode:
            continue
        env = diagnostics.get("environment_config", {})
        if list(env.get("near_shelter_capacity_candidates", [])) != near_capacities:
            continue
        if env.get("far_shelter_capacity") != far_capacity:
            continue
        if float(env.get("failure_penalty", float("nan"))) != failure_penalty:
            continue
        if env.get("shelter_capacity_mode") != capacity_mode.replace("-", "_"):
            continue
        expected_schema = {
            "absolute-and-demand-ratio": "capacity_absolute_and_ratio"
        }[capacity_observation]
        if env.get("observation_schema") != expected_schema:
            continue
        seed = int(diagnostics["seed"])
        previous = selected.get(seed)
        if previous is None or diagnostics_path.stat().st_mtime > previous[0].stat().st_mtime:
            selected[seed] = (diagnostics_path, diagnostics)
    return selected


def main() -> None:
    args = parse_args()
    search_paths = [args.output_dir, *args.include]
    runs = load_runs(
        search_paths, args.entropy_mode, args.near_capacities, args.far_capacity,
        args.failure_penalty, args.capacity_mode, args.capacity_observation,
    )
    if args.train:
        for seed in args.seeds:
            if seed in runs:
                print(f"Skipping seed {seed}: matching diagnostics already found at {runs[seed][0]}")
                continue
            command = [
                sys.executable, str(Path(__file__).with_name("training.py")),
                "--episodes", str(args.episodes), "--entropy-mode", args.entropy_mode,
                "--seed", str(seed), "--num-agents", str(args.num_agents),
                "--max-steps", str(args.max_steps), "--minibatch-size", str(args.minibatch_size),
                "--device", args.device, "--output-dir", str(args.output_dir),
                "--near-capacities", *[str(value) for value in args.near_capacities],
                "--far-capacity", str(args.far_capacity),
                "--failure-penalty", str(args.failure_penalty),
                "--capacity-mode", args.capacity_mode,
                "--capacity-observation", args.capacity_observation,
                "--log-every-updates", "25",
            ]
            print(f"\nTraining seed {seed}: {' '.join(command)}", flush=True)
            subprocess.run(command, check=True)

        runs = load_runs(
            search_paths, args.entropy_mode, args.near_capacities, args.far_capacity,
            args.failure_penalty, args.capacity_mode, args.capacity_observation,
        )
    missing = [seed for seed in args.seeds if seed not in runs]
    if missing:
        raise FileNotFoundError(
            f"No {args.entropy_mode} diagnostics found for seeds {missing}. "
            "Use --train or add their parent directories with --include."
        )
    runs = {seed: runs[seed] for seed in args.seeds}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    summary = []
    for seed, (path, diagnostics) in runs.items():
        episode = np.asarray(diagnostics["episode"])
        metrics = diagnostics["episode_metrics"]
        update_episode = np.asarray(diagnostics["update_episode"])
        return_values = np.asarray(metrics["mean_episode_return_per_agent"], dtype=float)
        arrival_values = np.asarray(metrics["mean_arrival_time_with_timeouts"], dtype=float)
        far_values = np.asarray(metrics["sampled_far_fraction"], dtype=float)
        failure_values = np.asarray(metrics["failure_count"], dtype=float)
        entropy_values = np.asarray(diagnostics["update_metrics"]["policy_entropy"], dtype=float)
        critic_values = np.asarray(diagnostics["update_metrics"]["critic_loss"], dtype=float)
        x, values = rolling_mean(return_values, args.rolling_window)
        axes[0, 0].plot(episode[x - 1], values, label=f"seed {seed}")
        x, values = rolling_mean(arrival_values, args.rolling_window)
        axes[0, 1].plot(episode[x - 1], values, label=f"seed {seed}")
        x, values = rolling_mean(far_values, args.rolling_window)
        axes[0, 2].plot(episode[x - 1], values, label=f"seed {seed}")
        x, values = rolling_mean(failure_values, args.rolling_window)
        axes[1, 0].plot(episode[x - 1], values, label=f"seed {seed}")
        update_window = max(1, args.rolling_window // 4)
        x, values = rolling_mean(entropy_values, update_window)
        axes[1, 1].plot(update_episode[x - 1], values, label=f"seed {seed}")
        x, values = rolling_mean(critic_values, update_window)
        axes[1, 2].plot(update_episode[x - 1], values, label=f"seed {seed}")
        width = min(args.rolling_window, len(return_values))
        summary.append({
            "seed": seed, "diagnostics": str(path.resolve()),
            "episodes": len(episode),
            "final_mean_return": float(np.mean(return_values[-width:])),
            "final_mean_arrival_all": float(np.mean(arrival_values[-width:])),
            "final_sampled_far_fraction": float(np.mean(far_values[-width:])),
            "final_failed_agents": float(np.mean(failure_values[-width:])),
            "final_policy_entropy": float(np.mean(entropy_values[-max(1, width // 4):])),
            "final_critic_loss": float(np.mean(critic_values[-max(1, width // 4):])),
        })
    titles = [
        "Mean episode return / agent", "Mean arrival time (all agents)",
        "Sampled far fraction", "Failed agents", "Policy entropy", "Critic loss",
    ]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0, 1].set_ylabel("Seconds")
    axes[0, 2].set_ylim(0, 1)
    axes[1, 1].axhline(np.log(2), color="black", linestyle="--", linewidth=0.8)
    axes[1, 2].set_yscale("log")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    png_path = args.output_dir / f"seed_convergence_{stamp}.png"
    json_path = args.output_dir / f"seed_convergence_{stamp}.json"
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    aggregate = {
        "seed_count": len(summary),
        "return_mean_across_seeds": float(np.mean([row["final_mean_return"] for row in summary])),
        "return_std_across_seeds": float(np.std([row["final_mean_return"] for row in summary])),
        "arrival_mean_across_seeds": float(np.mean([row["final_mean_arrival_all"] for row in summary])),
        "arrival_std_across_seeds": float(np.std([row["final_mean_arrival_all"] for row in summary])),
        "far_mean_across_seeds": float(np.mean([row["final_sampled_far_fraction"] for row in summary])),
        "far_std_across_seeds": float(np.std([row["final_sampled_far_fraction"] for row in summary])),
        "failure_mean_across_seeds": float(np.mean([row["final_failed_agents"] for row in summary])),
        "failure_std_across_seeds": float(np.std([row["final_failed_agents"] for row in summary])),
    }
    json_path.write_text(json.dumps({"runs": summary, "aggregate": aggregate}, indent=2), encoding="utf-8")
    for row in summary:
        print(
            f"seed={row['seed']} return={row['final_mean_return']:.3f} "
            f"arrival_all={row['final_mean_arrival_all']:.1f}s "
            f"sampled_far={row['final_sampled_far_fraction']:.3f} "
            f"failures={row['final_failed_agents']:.2f} "
            f"entropy={row['final_policy_entropy']:.4f} "
            f"critic_loss={row['final_critic_loss']:.6f}"
        )
    print(
        f"Across seeds: return={aggregate['return_mean_across_seeds']:.3f}"
        f"+/-{aggregate['return_std_across_seeds']:.3f}, "
        f"arrival={aggregate['arrival_mean_across_seeds']:.1f}"
        f"+/-{aggregate['arrival_std_across_seeds']:.1f}s"
    )
    print(f"Saved: {png_path.resolve()}")
    print(f"Saved: {json_path.resolve()}")


if __name__ == "__main__":
    main()
