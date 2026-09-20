"""Plot the normalized excess-congestion function used by the environment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RHO_FREE = 0.1
RHO_MAX = 1.0


def normalized_excess_congestion(density: np.ndarray) -> np.ndarray:
    return np.clip((density - RHO_FREE) / (RHO_MAX - RHO_FREE), 0.0, 1.0)


def main() -> None:
    density = np.linspace(0.0, 1.25, 501)
    congestion = normalized_excess_congestion(density)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(density, congestion, color="#2878b5", linewidth=3)
    ax.axvline(RHO_FREE, color="#4a4a4a", linestyle="--", linewidth=1.5)
    ax.axvline(RHO_MAX, color="#4a4a4a", linestyle="--", linewidth=1.5)
    ax.scatter(
        [RHO_FREE, RHO_MAX],
        [0.0, 1.0],
        color="#d9534f",
        s=55,
        zorder=3,
    )

    ax.annotate(
        r"free-flow threshold  $\rho_{free}=0.1$",
        xy=(RHO_FREE, 0.0),
        xytext=(0.18, 0.16),
        arrowprops={"arrowstyle": "->", "color": "#4a4a4a"},
    )
    ax.annotate(
        r"saturation reference  $\rho_{max}=1.0$",
        xy=(RHO_MAX, 1.0),
        xytext=(0.63, 0.78),
        arrowprops={"arrowstyle": "->", "color": "#4a4a4a"},
    )

    ax.text(0.045, 0.08, "no congestion penalty", ha="center", color="#555555")
    ax.text(0.55, 0.53, "linear increase", ha="center", color="#555555")
    ax.text(1.12, 0.92, "clipped at 1", ha="center", color="#555555")

    ax.set(
        title="Normalized excess congestion used by the environment",
        xlabel=r"Physical edge density  $\rho_e(t)$  (agents / m$^2$)",
        ylabel=r"Normalized excess congestion  $C_e(t)$",
        xlim=(0.0, 1.25),
        ylim=(-0.04, 1.08),
    )
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(alpha=0.25)
    fig.tight_layout()

    output = Path(__file__).with_name("normalized_excess_congestion.png")
    fig.savefig(output, dpi=180)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
