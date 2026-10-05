"""Save the latest checkpoint when its recorded validation clears robust margins."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--min-return-margin", type=float, default=0.5)
    parser.add_argument("--min-arrival-margin", type=float, default=1.0)
    args = parser.parse_args()
    run = args.run.resolve()
    with (run / "diagnostics.pkl").open("rb") as handle:
        diagnostics = pickle.load(handle)
    checkpoint = torch.load(
        run / "latest_checkpoint.pt", map_location="cpu", weights_only=False
    )
    validation = diagnostics["validation"][-1]
    if int(validation["episode"]) != int(checkpoint["episode_completed"]):
        raise ValueError("Latest validation and latest checkpoint episodes differ")
    baseline = diagnostics["joint_fixed_baseline"]
    return_margin = (
        validation["mean_episode_return_per_agent"]
        - baseline["mean_episode_return_per_agent"]
    )
    arrival_margin = (
        baseline["mean_arrival_time_with_timeouts"]
        - validation["mean_arrival_time_with_timeouts"]
    )
    eligible = (
        validation["failure_count"] <= baseline["failure_count"]
        and return_margin >= args.min_return_margin
        and arrival_margin >= args.min_arrival_margin
    )
    if not eligible:
        raise ValueError(
            f"Latest Actor is not robust-eligible: return margin={return_margin:.3f}, "
            f"arrival margin={arrival_margin:.3f}s"
        )
    output = run / "best_robust_joint_actor.pt"
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint["actor_state_dict"], temporary)
    temporary.replace(output)

    metadata_path = run / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "robust_joint_min_return_margin": args.min_return_margin,
        "robust_joint_min_arrival_margin": args.min_arrival_margin,
        "best_robust_joint_validation_return": validation[
            "mean_episode_return_per_agent"
        ],
        "best_robust_joint_validation_arrival": validation[
            "mean_arrival_time_with_timeouts"
        ],
        "best_robust_joint_episode": int(validation["episode"]),
        "has_best_robust_joint_actor": True,
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"Saved {output}: episode={validation['episode']} "
        f"return_margin={return_margin:.3f}, "
        f"arrival_margin={arrival_margin:.3f}s"
    )


if __name__ == "__main__":
    main()
