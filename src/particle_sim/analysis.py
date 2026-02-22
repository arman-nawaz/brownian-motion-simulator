import numpy as np


def mean_squared_displacement(traj: np.ndarray) -> np.ndarray:
    """
    Compute MSD for single trajectory.
    """
    traj = np.asarray(traj, dtype=float)
    if traj.ndim != 2 or traj.shape[1] != 3:
        raise ValueError("traj must have shape (N,3)")

    disp = traj - traj[0]
    return np.sum(disp * disp, axis=1)


def mean_squared_displacement_many(trajs: np.ndarray) -> np.ndarray:
    """
    Compute averaged MSD over many trajectories.
    """
    trajs = np.asarray(trajs, dtype=float)
    if trajs.ndim != 3 or trajs.shape[2] != 3:
        raise ValueError("trajs must have shape (P,N,3)")

    disp = trajs - trajs[:, 0:1, :]
    msd = np.sum(disp * disp, axis=2)
    return msd.mean(axis=0)


def msd_theory_3d(t: np.ndarray, diffusion: float) -> np.ndarray:
    """
    Theoretical MSD in 3D: MSD = 6 D t
    """
    if diffusion < 0:
        raise ValueError("diffusion must be >= 0")

    return 6.0 * diffusion * np.asarray(t)


def estimate_diffusion_from_msd(
    t: np.ndarray,
    msd: np.ndarray,
    fit_start: float,
) -> float:
    """
    Estimate D from linear fit: slope = 6D
    """
    t = np.asarray(t)
    msd = np.asarray(msd)

    mask = t >= fit_start
    if mask.sum() < 2:
        raise ValueError("Not enough points to fit")

    slope, _ = np.polyfit(t[mask], msd[mask], 1)
    return slope / 6.0