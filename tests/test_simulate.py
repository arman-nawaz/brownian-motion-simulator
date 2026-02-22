import numpy as np
from particle_sim.simulate import simulate_brownian_3d, simulate_many
from particle_sim.analysis import mean_squared_displacement


def test_seed_reproducible():
    a = simulate_brownian_3d(n_steps=50, dt=0.1, diffusion=1.0, drift=(0, 0, 0), seed=123)
    b = simulate_brownian_3d(n_steps=50, dt=0.1, diffusion=1.0, drift=(0, 0, 0), seed=123)
    assert np.allclose(a, b)


def test_msd_nonnegative():
    traj = simulate_brownian_3d(n_steps=100, dt=0.1, diffusion=1.0, drift=(0, 0, 0), seed=0)
    msd = mean_squared_displacement(traj)
    assert np.all(msd >= 0)


def test_many_shape():
    trajs = simulate_many(
        n_particles=10,
        n_steps=20,
        dt=0.1,
        diffusion=1.0,
        drift=(0, 0, 0),
        seed=1,
    )
    assert trajs.shape == (10, 21, 3)