# absice/runner.py

import os
import shutil
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ABM_CE_PV_Model import ABM_CE_PV
from absice.data.data_loader import DataLoader, LoadedData
from absice.schemas.data_paths_config import DataPathsConfig
from absice.schemas.simulation_config import SimulationConfig


def run_single(
    config_path: Path,
    paths_yaml: Path | None,
    output_dir: Path,
    label: str,
    project_root: Path,
) -> Path:
    """
    Run one ABSiCE simulation and save its results to CSV.

    Results are written to:

    ``output_dir / label / Results_model_run_<seed>.csv``

    Parameters
    ----------
    config_path
        Path to the simulation configuration YAML file.
    paths_yaml
        Optional path to a data-paths YAML file. When omitted, default
        project-relative paths are used.
    output_dir
        Base directory where simulation results are written.
    label
        Name of the scenario-specific output subdirectory.
    project_root
        Root directory of the ABSiCE repository.

    Returns
    -------
    Path
        Path to the CSV file created by the simulation.
    """
    config = SimulationConfig.from_yaml(config_path)

    paths = _resolve_paths(
        paths_yaml=paths_yaml,
        project_root=project_root,
    )

    data = DataLoader(paths).load_all(
        resolution=config.consumer.resolution,
        model_states=config.consumer.model_states,
    )

    run_id = config.run.seed if config.run.seed is not None else 0

    output_path = _make_output_path(
        output_dir=output_dir,
        label=label,
        run_id=run_id,
    )

    _execute_and_save(
        config=config,
        data=data,
        output_path=output_path,
    )

    print(
        f"Results written to {output_path}",
        flush=True,
    )

    return output_path


def run_batch(
    config_path: Path,
    paths_yaml: Path | None,
    n_runs: int,
    workers: int | None,
    output_dir: Path,
    label: str,
    project_root: Path,
) -> list[Path]:
    """
    Run independent ABSiCE replicates in parallel.

    Each replicate receives a unique seed based on its run ID. Every
    worker loads its own configuration and input data, runs the model,
    and writes a separate CSV file.

    Parameters
    ----------
    config_path
        Path to the simulation configuration YAML file.
    paths_yaml
        Optional path to a data-paths YAML file. When omitted, default
        project-relative paths are used.
    n_runs
        Number of independent simulation replicates.
    workers
        Number of parallel worker processes. When ``None``, Python uses
        the available CPU count.
    output_dir
        Base directory where batch results are written.
    label
        Name of the scenario-specific output subdirectory.
    project_root
        Root directory of the ABSiCE repository.

    Returns
    -------
    list[Path]
        Paths to all output CSV files, ordered by run ID.

    Raises
    ------
    ValueError
        If ``n_runs`` or ``workers`` is less than one.
    Exception
        Re-raises an exception if any worker process fails.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1.")

    if workers is not None and workers < 1:
        raise ValueError("workers must be at least 1.")

    scenario_dir = output_dir / label
    scenario_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    worker_count = workers or os.cpu_count() or 1
    start_time = time.perf_counter()

    print(
        f"Launching {n_runs} runs on {worker_count} workers...",
        flush=True,
    )

    futures_to_run_id: dict[Future[Path], int] = {}
    output_paths: list[Path | None] = [None] * n_runs

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for run_id in range(n_runs):
            future = pool.submit(
                _worker,
                run_id,
                config_path,
                paths_yaml,
                output_dir,
                label,
                project_root,
            )

            futures_to_run_id[future] = run_id

        for future in as_completed(futures_to_run_id):
            run_id = futures_to_run_id[future]

            try:
                result_path = future.result()
                output_paths[run_id] = result_path

                print(
                    f"Run {run_id} completed: {result_path}",
                    flush=True,
                )

            except Exception as error:
                print(
                    f"Run {run_id} FAILED: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

    elapsed = time.perf_counter() - start_time

    print(
        f"All {n_runs} runs completed in {elapsed:.1f} seconds.",
        flush=True,
    )

    return [
        path
        for path in output_paths
        if path is not None
    ]


def write_default_config(output: Path) -> None:
    """
    Write a default simulation configuration YAML file.

    If ``config/default_simulation.yaml`` exists, it is copied so any
    existing comments and formatting are preserved. Otherwise, a new
    configuration is generated from the current ``SimulationConfig``
    defaults.

    Parameters
    ----------
    output
        Destination path for the new YAML configuration file.
    """
    output = Path(output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    project_root = Path(__file__).resolve().parent.parent
    default_yaml = (
        project_root
        / "config"
        / "default_simulation.yaml"
    )

    if default_yaml.exists():
        source = default_yaml.resolve()
        destination = output.resolve()

        if source == destination:
            return

        shutil.copyfile(
            source,
            destination,
        )
        return

    config = SimulationConfig()
    config.to_yaml(output)


def _worker(
    run_id: int,
    config_path: Path,
    paths_yaml: Path | None,
    output_dir: Path,
    label: str,
    project_root: Path,
) -> Path:
    """
    Execute one batch replicate in a worker process.

    This function must remain at module level so that it can be pickled
    and executed by ``ProcessPoolExecutor``.

    Parameters
    ----------
    run_id
        Replicate number used as the simulation seed and output filename.
    config_path
        Path to the simulation configuration YAML file.
    paths_yaml
        Optional path to a data-paths YAML file.
    output_dir
        Base results directory.
    label
        Scenario-specific output subdirectory.
    project_root
        Root directory of the ABSiCE repository.

    Returns
    -------
    Path
        Path to the CSV file created by this worker.
    """
    config = SimulationConfig.from_yaml(config_path)

    run_config = config.run.model_copy(
        update={"seed": run_id},
    )

    config = config.model_copy(
        update={"run": run_config},
    )

    paths = _resolve_paths(
        paths_yaml=paths_yaml,
        project_root=project_root,
    )

    data = DataLoader(paths).load_all(
        resolution=config.consumer.resolution,
        model_states=config.consumer.model_states,
    )

    output_path = _make_output_path(
        output_dir=output_dir,
        label=label,
        run_id=run_id,
    )

    _execute_and_save(
        config=config,
        data=data,
        output_path=output_path,
    )

    return output_path


def _resolve_paths(
    paths_yaml: Path | None,
    project_root: Path,
) -> DataPathsConfig:
    """
    Load a data-path configuration or construct the default paths.

    Parameters
    ----------
    paths_yaml
        Optional path to a data-paths YAML file.
    project_root
        Root directory used to resolve relative paths.

    Returns
    -------
    DataPathsConfig
        Validated paths for all external input data.
    """
    if paths_yaml is not None:
        return DataPathsConfig.from_yaml(
            path=paths_yaml,
            project_root=project_root,
        )

    return DataPathsConfig.make_default(project_root)


def _make_output_path(
    output_dir: Path,
    label: str,
    run_id: int,
) -> Path:
    """
    Create a scenario output directory and return its CSV path.

    Parameters
    ----------
    output_dir
        Base results directory.
    label
        Scenario-specific subdirectory name.
    run_id
        Number used in the output filename.

    Returns
    -------
    Path
        Output path in the form
        ``output_dir/label/Results_model_run_<run_id>.csv``.
    """
    destination = output_dir / label
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    return destination / f"Results_model_run_{run_id}.csv"


def _execute_and_save(
    config: SimulationConfig,
    data: LoadedData,
    output_path: Path,
) -> None:
    """
    Instantiate the model, run all steps, and save model-level results.

    Parameters
    ----------
    config
        Validated simulation configuration.
    data
        All datasets loaded by ``DataLoader``.
    output_path
        CSV destination for the model-level DataCollector output.
    """
    model = ABM_CE_PV(
        config=config,
        data=data,
    )

    for _ in range(config.run.last_step):
        model.step()

    results: pd.DataFrame = (
        model.datacollector.get_model_vars_dataframe()
    )

    results.to_csv(output_path)