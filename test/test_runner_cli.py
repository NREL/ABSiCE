"""Tests for the ABSiCE command-line interface."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from run import cli


def test_cli_registers_expected_commands() -> None:
    """The CLI exposes the run, batch, and init-config commands."""
    assert set(cli.commands.keys()) == {
        "run",
        "batch",
        "init-config",
    }


def test_run_delegates_to_run_single(
    tmp_path: Path,
) -> None:
    """The run command passes the expected arguments to run_single."""
    config_path = tmp_path / "sim.yaml"
    config_path.write_text(
        "{}",
        encoding="utf-8",
    )

    output_dir = tmp_path / "results"
    runner = CliRunner()

    with patch(
        "absice.runner.run_single"
    ) as mock_run_single:
        result = runner.invoke(
            cli,
            [
                "run",
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    mock_run_single.assert_called_once()

    call_kwargs = mock_run_single.call_args.kwargs

    assert call_kwargs["config_path"] == config_path
    assert call_kwargs["paths_yaml"] is None
    assert call_kwargs["output_dir"] == output_dir
    assert call_kwargs["label"] == "sim"
    assert call_kwargs["project_root"].name != ""


def test_run_uses_custom_label(
    tmp_path: Path,
) -> None:
    """The run command passes a custom output label when supplied."""
    config_path = tmp_path / "sim.yaml"
    config_path.write_text(
        "{}",
        encoding="utf-8",
    )

    runner = CliRunner()

    with patch(
        "absice.runner.run_single"
    ) as mock_run_single:
        result = runner.invoke(
            cli,
            [
                "run",
                "--config",
                str(config_path),
                "--label",
                "baseline",
            ],
        )

    assert result.exit_code == 0, result.output
    assert (
        mock_run_single.call_args.kwargs["label"]
        == "baseline"
    )


def test_batch_delegates_to_run_batch(
    tmp_path: Path,
) -> None:
    """The batch command passes batch options to run_batch."""
    config_path = tmp_path / "batch_config.yaml"
    config_path.write_text(
        "{}",
        encoding="utf-8",
    )

    output_dir = tmp_path / "batch_results"
    runner = CliRunner()

    with patch(
        "absice.runner.run_batch"
    ) as mock_run_batch:
        result = runner.invoke(
            cli,
            [
                "batch",
                "--config",
                str(config_path),
                "--n-runs",
                "5",
                "--workers",
                "2",
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    mock_run_batch.assert_called_once()

    call_kwargs = mock_run_batch.call_args.kwargs

    assert call_kwargs["config_path"] == config_path
    assert call_kwargs["paths_yaml"] is None
    assert call_kwargs["n_runs"] == 5
    assert call_kwargs["workers"] == 2
    assert call_kwargs["output_dir"] == output_dir
    assert call_kwargs["label"] == "batch_config"


def test_batch_uses_custom_paths_and_label(
    tmp_path: Path,
) -> None:
    """The batch command accepts custom paths and output labels."""
    config_path = tmp_path / "scenario.yaml"
    config_path.write_text(
        "{}",
        encoding="utf-8",
    )

    paths_path = tmp_path / "data_paths.yaml"
    paths_path.write_text(
        "{}",
        encoding="utf-8",
    )

    runner = CliRunner()

    with patch(
        "absice.runner.run_batch"
    ) as mock_run_batch:
        result = runner.invoke(
            cli,
            [
                "batch",
                "--config",
                str(config_path),
                "--paths",
                str(paths_path),
                "--label",
                "scenario_a",
                "--n-runs",
                "3",
            ],
        )

    assert result.exit_code == 0, result.output

    call_kwargs = mock_run_batch.call_args.kwargs

    assert call_kwargs["paths_yaml"] == paths_path
    assert call_kwargs["label"] == "scenario_a"
    assert call_kwargs["n_runs"] == 3


def test_init_config_delegates_to_write_default_config(
    tmp_path: Path,
) -> None:
    """The init-config command delegates to write_default_config."""
    output_path = tmp_path / "new_config.yaml"
    runner = CliRunner()

    with patch(
        "absice.runner.write_default_config"
    ) as mock_write:
        result = runner.invoke(
            cli,
            [
                "init-config",
                "--output",
                str(output_path),
            ],
        )

    assert result.exit_code == 0, result.output
    mock_write.assert_called_once_with(output_path)

    assert (
        f"Default configuration written to {output_path}"
        in result.output
    )


def test_run_rejects_missing_config_file(
    tmp_path: Path,
) -> None:
    """The run command rejects a configuration path that does not exist."""
    missing_path = tmp_path / "missing.yaml"
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "run",
            "--config",
            str(missing_path),
        ],
    )

    assert result.exit_code != 0
    assert "does not exist" in result.output.lower()


def test_batch_rejects_zero_runs(
    tmp_path: Path,
) -> None:
    """The batch command rejects an n-runs value below one."""
    config_path = tmp_path / "sim.yaml"
    config_path.write_text(
        "{}",
        encoding="utf-8",
    )

    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "batch",
            "--config",
            str(config_path),
            "--n-runs",
            "0",
        ],
    )

    assert result.exit_code != 0