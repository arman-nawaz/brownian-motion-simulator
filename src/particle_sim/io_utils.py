import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class SimConfig:
    n_steps: int
    dt: float
    diffusion: float
    drift: tuple[float, float, float]
    seed: int
    n_particles: int
    fit_start: float


def load_config(path: str) -> SimConfig:
    data = json.loads(Path(path).read_text())

    return SimConfig(
        n_steps=data["n_steps"],
        dt=data["dt"],
        diffusion=data["diffusion"],
        drift=tuple(data["drift"]),
        seed=data["seed"],
        n_particles=data["n_particles"],
        fit_start=data["fit_start"],
    )


def save_csv(path: Path, header: str, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, delimiter=",", header=header, comments="")