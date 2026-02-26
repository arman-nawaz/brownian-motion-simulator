import numpy as np
import pytest

from particle_sim.simulate import simulate_brownian_3d, simulate_many
from particle_sim.analysis import (
    mean_squared_displacement,
    mean_squared_displacement_many,
    msd_theory_3d,
    estimate_diffusion_from_msd,
)


# -------------------------
# Simulation tests
# -------------------------

def test_seed_reproducible():
    a = simulate_brownian_3d(n_steps=50, dt=0.1, diffusion=1.0, drift=(0, 0, 0), seed=123)
    b = simulate_brownian_3d(n_steps=50, dt=0.1, diffusion=1.0, drift=(0, 0, 0), seed=123)
    assert np.allclose(a, b)


def test_shape_and_dtype_single():
    traj = simulate_brownian_3d(n_steps=10, dt=0.1, diffusion=1.0, drift=(0, 0, 0), seed=0)
    assert traj.shape == (11, 3)
    assert traj.dtype.kind == "f"


def test_invalid_inputs_single():
    with pytest.raises(ValueError):
        simulate_brownian_3d(n_steps=0, dt=0.1, diffusion=1.0, drift=(0, 0, 0), seed=0)
    with pytest.raises(ValueError):
        simulate_brownian_3d(n_steps=10, dt=0.0, diffusion=1.0, drift=(0, 0, 0), seed=0)
    with pytest.raises(ValueError):
        simulate_brownian_3d(n_steps=10, dt=0.1, diffusion=-1.0, drift=(0, 0, 0), seed=0)
    with pytest.raises(ValueError):
        simulate_brownian_3d(n_steps=10, dt=0.1, diffusion=1.0, drift=(0, 0), seed=0)


def test_zero_motion_when_no_diffusion_and_no_drift():
    traj = simulate_brownian_3d(n_steps=20, dt=0.1, diffusion=0.0, drift=(0, 0, 0), seed=1)
    assert np.allclose(traj, 0.0)


def test_drift_only_is_linear_motion():
    dt = 0.5
    drift = (2.0, -1.0, 0.25)  # velocity
    n_steps = 10

    traj = simulate_brownian_3d(n_steps=n_steps, dt=dt, diffusion=0.0, drift=drift, seed=0)
    t = np.arange(n_steps + 1) * dt
    expected = np.stack([drift[0] * t, drift[1] * t, drift[2] * t], axis=1)
    assert np.allclose(traj, expected)


def test_many_shape_and_reproducible():
    trajs1 = simulate_many(n_particles=5, n_steps=7, dt=0.1, diffusion=1.0, drift=(0, 0, 0), seed=123)
    trajs2 = simulate_many(n_particles=5, n_steps=7, dt=0.1, diffusion=1.0, drift=(0, 0, 0), seed=123)
    assert trajs1.shape == (5, 8, 3)
    assert np.allclose(trajs1, trajs2)


def test_invalid_inputs_many():
    with pytest.raises(ValueError):
        simulate_many(n_particles=0, n_steps=10, dt=0.1, diffusion=1.0, drift=(0, 0, 0), seed=0)
    with pytest.raises(ValueError):
        simulate_many(n_particles=10, n_steps=0, dt=0.1, diffusion=1.0, drift=(0, 0, 0), seed=0)
    with pytest.raises(ValueError):
        simulate_many(n_particles=10, n_steps=10, dt=0.0, diffusion=1.0, drift=(0, 0, 0), seed=0)
    with pytest.raises(ValueError):
        simulate_many(n_particles=10, n_steps=10, dt=0.1, diffusion=-1.0, drift=(0, 0, 0), seed=0)
    with pytest.raises(ValueError):
        simulate_many(n_particles=10, n_steps=10, dt=0.1, diffusion=1.0, drift=(0, 0), seed=0)


# -------------------------
# Analysis tests
# -------------------------

def test_msd_nonnegative():
    traj = simulate_brownian_3d(n_steps=100, dt=0.1, diffusion=1.0, drift=(0, 0, 0), seed=0)
    msd = mean_squared_displacement(traj)
    assert np.all(msd >= 0)


def test_msd_known_linear_trajectory():
    # r(t) = (t,0,0) -> MSD = t^2 (since r(0)=0)
    t = np.arange(6, dtype=float)
    traj = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    msd = mean_squared_displacement(traj)
    assert np.allclose(msd, t**2)


def test_msd_many_matches_single_when_particles_equal():
    traj = simulate_brownian_3d(n_steps=30, dt=0.1, diffusion=1.0, drift=(0, 0, 0), seed=10)
    trajs = traj[None, :, :]  # shape (1,N,3)

    msd_single = mean_squared_displacement(traj)
    msd_many = mean_squared_displacement_many(trajs)
    assert np.allclose(msd_single, msd_many)


def test_msd_theory_3d():
    t = np.array([0.0, 1.0, 2.5])
    D = 0.7
    expected = 6.0 * D * t
    assert np.allclose(msd_theory_3d(t, D), expected)

    with pytest.raises(ValueError):
        msd_theory_3d(t, -1.0)


def test_estimate_diffusion_on_perfect_theory_data():
    D_true = 1.25
    t = np.linspace(0, 10, 200)
    msd = 6.0 * D_true * t
    D_est = estimate_diffusion_from_msd(t, msd, fit_start=0.0)
    assert np.isclose(D_est, D_true, rtol=1e-6, atol=1e-10)


def test_estimate_diffusion_from_simulation_average_reasonable():
    # noisy but should be in the right range if we average many trajectories
    D_true = 1.0
    dt = 0.1
    n_steps = 400
    n_particles = 400

    trajs = simulate_many(
        n_particles=n_particles,
        n_steps=n_steps,
        dt=dt,
        diffusion=D_true,
        drift=(0, 0, 0),
        seed=42,
    )
    t = np.arange(n_steps + 1) * dt
    msd_avg = mean_squared_displacement_many(trajs)

    D_est = estimate_diffusion_from_msd(t, msd_avg, fit_start=5.0)

    # allow tolerance (stochastic)
    assert np.isclose(D_est, D_true, rtol=0.25)