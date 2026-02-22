import numpy as np


def simulate_brownian_3d(
    n_steps: int,
    dt: float,
    diffusion: float,
    drift: tuple[float, float, float],
    seed: int,
) -> np.ndarray:
    """
    Simulate 3D Brownian motion with optional drift.

    Returns
    -------
    traj : np.ndarray
        Shape (n_steps+1, 3)
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if diffusion < 0:
        raise ValueError("diffusion must be >= 0")

    drift = np.asarray(drift, dtype=float)
    if drift.shape != (3,):
        raise ValueError("drift must be length 3")

    rng = np.random.default_rng(seed)

    sigma = np.sqrt(2.0 * diffusion * dt)
    noise = rng.normal(0.0, sigma, size=(n_steps, 3))
    steps = noise + drift * dt

    traj = np.zeros((n_steps + 1, 3))
    traj[1:] = np.cumsum(steps, axis=0)
    return traj


def simulate_many(
    n_particles: int,
    n_steps: int,
    dt: float,
    diffusion: float,
    drift: tuple[float, float, float],
    seed: int,
) -> np.ndarray:
    """
    Simulate multiple independent trajectories.

    Returns
    -------
    trajs : np.ndarray
        Shape (n_particles, n_steps+1, 3)
    """
    if n_particles <= 0:
        raise ValueError("n_particles must be > 0")

    rng = np.random.default_rng(seed)
    trajs = np.zeros((n_particles, n_steps + 1, 3))

    for i in range(n_particles):
        trajs[i] = simulate_brownian_3d(
            n_steps=n_steps,
            dt=dt,
            diffusion=diffusion,
            drift=drift,
            seed=int(rng.integers(0, 2**31 - 1)),
        )

    return trajs