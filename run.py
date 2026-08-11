"""
ABSiCE simulation runner.

Usage examples
--------------
Run one simulation:
    python run.py run

Run one simulation with a custom configuration:
    python run.py run --config config/my_scenario.yaml

Run ten simulations using four workers:
    python run.py batch --n-runs 10 --workers 4

Create a new configuration file:
    python run.py init-config --output config/my_scenario.yaml
"""

from pathlib import Path

import click


PROJECT_ROOT: Path = Path(__file__).parent


@click.group()
def cli() -> None:
    """ABSiCE command-line interface."""


@cli.command()
@click.option(
    "--config",
    "-c",
    default="config/default_simulation.yaml",
    show_default=True,
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    help="Path to the simulation configuration YAML.",
)
@click.option(
    "--paths",
    "-p",
    default=None,
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    help="Optional path to the data-paths YAML file.",
)
@click.option(
    "--output-dir",
    "-o",
    default="results",
    show_default=True,
    type=click.Path(
        file_okay=False,
        path_type=Path,
    ),
    help="Base directory where results will be written.",
)
@click.option(
    "--label",
    "-l",
    default=None,
    help=(
        "Output subfolder name. Defaults to the configuration "
        "file name without the extension."
    ),
)
def run(
    config: Path,
    paths: Path | None,
    output_dir: Path,
    label: str | None,
) -> None:
    """Run one simulation."""
    from absice.runner import run_single

    run_single(
        config_path=config,
        paths_yaml=paths,
        output_dir=output_dir,
        label=label or config.stem,
        project_root=PROJECT_ROOT,
    )


@cli.command()
@click.option(
    "--config",
    "-c",
    default="config/default_simulation.yaml",
    show_default=True,
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    help="Path to the simulation configuration YAML.",
)
@click.option(
    "--paths",
    "-p",
    default=None,
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    help="Optional path to the data-paths YAML file.",
)
@click.option(
    "--n-runs",
    "-n",
    default=10,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of independent simulations to run.",
)
@click.option(
    "--workers",
    "-w",
    default=None,
    type=click.IntRange(min=1),
    help="Number of worker processes. Defaults to available CPU cores.",
)
@click.option(
    "--output-dir",
    "-o",
    default="results",
    show_default=True,
    type=click.Path(
        file_okay=False,
        path_type=Path,
    ),
    help="Base directory where results will be written.",
)
@click.option(
    "--label",
    "-l",
    default=None,
    help=(
        "Output subfolder name. Defaults to the configuration "
        "file name without the extension."
    ),
)
def batch(
    config: Path,
    paths: Path | None,
    n_runs: int,
    workers: int | None,
    output_dir: Path,
    label: str | None,
) -> None:
    """Run multiple simulations in parallel."""
    from absice.runner import run_batch

    run_batch(
        config_path=config,
        paths_yaml=paths,
        n_runs=n_runs,
        workers=workers,
        output_dir=output_dir,
        label=label or config.stem,
        project_root=PROJECT_ROOT,
    )


@cli.command("init-config")
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(
        dir_okay=False,
        path_type=Path,
    ),
    help="Path where the new configuration YAML will be written.",
)
def init_config(output: Path) -> None:
    """Create a copy of the default simulation configuration."""
    from absice.runner import write_default_config

    write_default_config(output)
    click.echo(f"Default configuration written to {output}")


if __name__ == "__main__":
    cli()