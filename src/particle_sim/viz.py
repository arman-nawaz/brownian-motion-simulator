import matplotlib.pyplot as plt
import numpy as np


def plot_msd(
    t: np.ndarray,
    msd: np.ndarray,
    msd_theory: np.ndarray | None = None,
    title: str = "MSD (3D)",
):
    """
    Plot MSD curve (and optional theory curve). Visualization only.
    Returns a matplotlib Figure object.
    """
    fig = plt.figure()
    plt.plot(t, msd, label="MSD (sim)")

    if msd_theory is not None:
        plt.plot(t, msd_theory, linestyle="--", label="MSD (theory)")

    plt.xlabel("Time")
    plt.ylabel("MSD")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return fig