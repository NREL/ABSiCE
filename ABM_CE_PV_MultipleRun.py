# -*- coding: utf-8 -*-
"""Run one or more ABSiCE simulations from a YAML configuration."""

from pathlib import Path

from ABM_CE_PV_Model import ABM_CE_PV
from absice.data.data_loader import DataLoader
from absice.schemas.data_paths_config import DataPathsConfig
from absice.schemas.simulation_config import SimulationConfig


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "default_simulation.yaml"
)


def run(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> None:
    """Load configuration and data, then run the ABSiCE model.

    Parameters
    ----------
    config_path
        Path to the simulation configuration YAML file.
    """
    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    print("Loading configuration...")
    config = SimulationConfig.from_yaml(config_path)

    print("Loading data...")
    paths = DataPathsConfig.make_default(
        project_root=PROJECT_ROOT,
    )

    data = DataLoader(paths).load_all(
        resolution=config.consumer.resolution,
        model_states=config.consumer.model_states,
    )

    print("Creating model...")
    model = ABM_CE_PV(
        config=config,
        data=data,
    )

    print("Running simulation...")
    for _ in range(config.run.last_step):
        model.step()

    print("Simulation complete.")


if __name__ == "__main__":
    run()