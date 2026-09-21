"""Plot the density-dependent walking-speed rule used by the environment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RHO_FREE = 0.1
RHO_MAX = 1.0
MIN_SPEED_FACTOR = 0.3
BASE_SPEEDS = (1.0, 1.25, 1.5)


def speed_factor(density: np.ndarray) -> np.ndarray:
    congestion = np.clip(
        (density - RHO_FREE) / (RHO_MAX - RHO_FREE), 0.0, 1.0
    )
    return 1.0 - (1.0 - MIN_SPEED_FACTOR) * congestion


def main() -> None:
    density = np.linspace(0.0, 1.25, 501)
    factor = speed_factor(density)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    colors = ("#2878b5", "#e17c05", "#3a923a")
    for base_speed, color in zip(BASE_SPEEDS, colors):
        ax.plot(
            density,
            base_speed * factor,
            linewidth=2.7,
            color=color,
            label=fr"base speed $v_0={base_speed:g}$ m/s",
        )

    ax.axvline(RHO_FREE, color="#555555", linestyle="--", linewidth=1.4)
    ax.axvline(RHO_MAX, color="#555555", linestyle="--", linewidth=1.4)
    ax.annotate(
        r"free-flow threshold  $\rho_{free}=0.1$",
        xy=(RHO_FREE, 1.0),
        xytext=(0.19, 1.11),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )
    ax.annotate(
        r"minimum speed factor $=0.3$ from $\rho_{max}=1.0$",
        xy=(RHO_MAX, 1.25 * MIN_SPEED_FACTOR),
        xytext=(0.56, 0.61),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )

    ax.set(
        title="Walking speed as a function of edge density",
        xlabel=r"Physical edge density  $\rho_e(t)$  (agents / m$^2$)",
        ylabel=r"Walking speed  $v_i(t)$  (m/s)",
        xlim=(0.0, 1.25),
        ylim=(0.2, 1.58),
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    output = Path(__file__).with_name("speed_density_relationship.png")
    fig.savefig(output, dpi=180)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
