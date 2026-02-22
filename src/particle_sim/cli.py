import argparse
from pathlib import Path
import numpy as np

from particle_sim.io_utils import load_config, save_csv
from particle_sim.simulate import simulate_brownian_3d, simulate_many
from particle_sim.analysis import (
    mean_squared_displacement,
    mean_squared_displacement_many,
    msd_theory_3d,
    estimate_diffusion_from_msd,
)
from particle_sim.viz import plot_msd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="3D Brownian Motion Simulator (Drift-Diffusion)")
    parser.add_argument("--config", default="configs/default.json", help="Path to config JSON")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--no-plot", action="store_true", help="Disable interactive plot window")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Time array
    t = np.arange(cfg.n_steps + 1) * cfg.dt

    # ---- Single trajectory ----
    traj = simulate_brownian_3d(
        n_steps=cfg.n_steps,
        dt=cfg.dt,
        diffusion=cfg.diffusion,
        drift=cfg.drift,
        seed=cfg.seed,
    )
    msd_single = mean_squared_displacement(traj)

    save_csv(outdir / "trajectory_single.csv", "t,x,y,z", np.c_[t, traj])
    save_csv(outdir / "msd_single.csv", "t,msd", np.c_[t, msd_single])

    # ---- Many trajectories (average MSD) ----
    trajs = simulate_many(
        n_particles=cfg.n_particles,
        n_steps=cfg.n_steps,
        dt=cfg.dt,
        diffusion=cfg.diffusion,
        drift=cfg.drift,
        seed=cfg.seed,
    )
    msd_avg = mean_squared_displacement_many(trajs)
    save_csv(outdir / "msd_avg.csv", "t,msd_avg", np.c_[t, msd_avg])

    # ---- Theory + estimate diffusion ----
    msd_th = msd_theory_3d(t, cfg.diffusion)
    d_est = estimate_diffusion_from_msd(t, msd_avg, fit_start=cfg.fit_start)

    # Save summary text
    (outdir / "summary.txt").write_text(
        f"config={args.config}\n"
        f"n_particles={cfg.n_particles}\n"
        f"n_steps={cfg.n_steps}\n"
        f"dt={cfg.dt}\n"
        f"D_true={cfg.diffusion}\n"
        f"D_est={d_est}\n",
        encoding="utf-8",
    )

    # Plot saved always
    fig = plot_msd(t, msd_avg, msd_th, title=f"MSD (avg) | D_est={d_est:.3f}")
    fig.savefig(outdir / "msd_compare.png", dpi=200)

    if not args.no_plot:
        import matplotlib.pyplot as plt
        plt.show()

    print(f"Done. Outputs saved to: {outdir.resolve()}")
    print(f"Estimated diffusion: D_est = {d_est:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())