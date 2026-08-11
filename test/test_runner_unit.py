"""Unit tests for ABSiCE runner functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import math

from absice.runner import (
    _execute_and_save,
    _make_output_path,
    _resolve_paths,
    run_batch,
    run_single,
    write_default_config,
)
from absice.schemas.simulation_config import SimulationConfig


def test_make_output_path_creates_directory(
    tmp_path: Path,
) -> None:
    """The output helper creates the scenario folder."""
    output_path = _make_output_path(
        output_dir=tmp_path,
        label="baseline",
        run_id=3,
    )

    expected_path = (
        tmp_path
        / "baseline"
        / "Results_model_run_3.csv"
    )

    assert output_path == expected_path
    assert output_path.parent.exists()
    assert output_path.parent.is_dir()


def test_resolve_paths_uses_default_paths(
    tmp_path: Path,
) -> None:
    """Default data paths are used when no YAML path is provided."""
    expected_paths = MagicMock()

    with patch(
        "absice.runner.DataPathsConfig.make_default",
        return_value=expected_paths,
    ) as mock_default:
        result = _resolve_paths(
            paths_yaml=None,
            project_root=tmp_path,
        )

    assert result is expected_paths
    mock_default.assert_called_once_with(tmp_path)


def test_resolve_paths_loads_yaml(
    tmp_path: Path,
) -> None:
    """A provided paths YAML is loaded relative to the project root."""
    paths_yaml = tmp_path / "data_paths.yaml"
    paths_yaml.write_text(
        "{}",
        encoding="utf-8",
    )

    expected_paths = MagicMock()

    with patch(
        "absice.runner.DataPathsConfig.from_yaml",
        return_value=expected_paths,
    ) as mock_from_yaml:
        result = _resolve_paths(
            paths_yaml=paths_yaml,
            project_root=tmp_path,
        )

    assert result is expected_paths

    mock_from_yaml.assert_called_once_with(
        path=paths_yaml,
        project_root=tmp_path,
    )


def test_execute_and_save_runs_expected_steps(
    tmp_path: Path,
) -> None:
    """The execution helper steps the model and writes its results."""
    output_path = tmp_path / "results.csv"

    config = SimulationConfig()
    config.run.last_step = 3

    data = MagicMock()

    results_dataframe = pd.DataFrame(
        {
            "Year": [2020, 2021, 2022],
            "Waste": [1.0, 2.0, 3.0],
        }
    )

    mock_model = MagicMock()
    mock_model.datacollector.get_model_vars_dataframe.return_value = (
        results_dataframe
    )

    with patch(
        "absice.runner.ABM_CE_PV",
        return_value=mock_model,
    ) as mock_model_class:
        _execute_and_save(
            config=config,
            data=data,
            output_path=output_path,
        )

    mock_model_class.assert_called_once_with(
        config=config,
        data=data,
    )

    assert mock_model.step.call_count == 3
    assert output_path.exists()

    saved_results = pd.read_csv(
        output_path,
        index_col=0,
    )

    pd.testing.assert_frame_equal(
        saved_results,
        results_dataframe,
    )


def test_run_single_loads_data_and_saves_results(
    tmp_path: Path,
) -> None:
    """run_single coordinates config loading, data loading, and execution."""
    config_path = tmp_path / "scenario.yaml"
    config_path.write_text(
        "{}",
        encoding="utf-8",
    )

    output_dir = tmp_path / "results"
    project_root = tmp_path

    config = SimulationConfig()
    config.run.seed = 42

    mock_paths = MagicMock()
    mock_data = MagicMock()
    expected_output = (
        output_dir
        / "baseline"
        / "Results_model_run_42.csv"
    )

    mock_loader = MagicMock()
    mock_loader.load_all.return_value = mock_data

    with (
        patch(
            "absice.runner.SimulationConfig.from_yaml",
            return_value=config,
        ) as mock_config_loader,
        patch(
            "absice.runner._resolve_paths",
            return_value=mock_paths,
        ) as mock_resolve,
        patch(
            "absice.runner.DataLoader",
            return_value=mock_loader,
        ) as mock_data_loader_class,
        patch(
            "absice.runner._execute_and_save",
        ) as mock_execute,
    ):
        result = run_single(
            config_path=config_path,
            paths_yaml=None,
            output_dir=output_dir,
            label="baseline",
            project_root=project_root,
        )

    assert result == expected_output

    mock_config_loader.assert_called_once_with(
        config_path
    )

    mock_resolve.assert_called_once_with(
        paths_yaml=None,
        project_root=project_root,
    )

    mock_data_loader_class.assert_called_once_with(
        mock_paths
    )

    mock_loader.load_all.assert_called_once_with(
        resolution=config.consumer.resolution,
        model_states=config.consumer.model_states,
    )

    mock_execute.assert_called_once_with(
        config=config,
        data=mock_data,
        output_path=expected_output,
    )


def test_run_single_uses_zero_when_seed_is_none(
    tmp_path: Path,
) -> None:
    """A single run uses run ID zero when no seed is configured."""
    config = SimulationConfig()
    config.run.seed = None

    mock_loader = MagicMock()
    mock_loader.load_all.return_value = MagicMock()

    with (
        patch(
            "absice.runner.SimulationConfig.from_yaml",
            return_value=config,
        ),
        patch(
            "absice.runner._resolve_paths",
            return_value=MagicMock(),
        ),
        patch(
            "absice.runner.DataLoader",
            return_value=mock_loader,
        ),
        patch(
            "absice.runner._execute_and_save",
        ),
    ):
        result = run_single(
            config_path=tmp_path / "config.yaml",
            paths_yaml=None,
            output_dir=tmp_path,
            label="test",
            project_root=tmp_path,
        )

    assert result.name == "Results_model_run_0.csv"


def test_write_default_config_generates_from_schema(
    tmp_path: Path,
) -> None:
    """A default config is generated when no template YAML exists."""
    output_path = tmp_path / "generated.yaml"

    with patch(
        "absice.runner.Path.exists",
        return_value=False,
    ):
        write_default_config(output_path)

    assert output_path.exists()

    loaded_config = SimulationConfig.from_yaml(output_path)
    default_config = SimulationConfig()

    assert loaded_config.run == default_config.run
    assert loaded_config.calibration == default_config.calibration
    assert loaded_config.network == default_config.network
    assert loaded_config.consumer == default_config.consumer
    assert loaded_config.product == default_config.product
    assert loaded_config.eol == default_config.eol
    assert loaded_config.tpb == default_config.tpb
    assert loaded_config.cost == default_config.cost
    assert loaded_config.scenario == default_config.scenario
    assert loaded_config.tclp == default_config.tclp
    assert loaded_config.data_source == default_config.data_source
    assert loaded_config.legacy_data == default_config.legacy_data

    # Verify NaN values were preserved
    assert math.isnan(
        loaded_config.material.scd_mat_prices["Product"][0]
    )
    assert math.isnan(
        loaded_config.material.virgin_mat_prices["Product"][0]
    )
    assert math.isnan(
        loaded_config.material.recovery_fractions["Product"]
    )


def test_run_batch_rejects_zero_runs(
    tmp_path: Path,
) -> None:
    """run_batch rejects fewer than one replicate."""
    with pytest.raises(
        ValueError,
        match="n_runs must be at least 1",
    ):
        run_batch(
            config_path=tmp_path / "config.yaml",
            paths_yaml=None,
            n_runs=0,
            workers=1,
            output_dir=tmp_path / "results",
            label="baseline",
            project_root=tmp_path,
        )


def test_run_batch_rejects_zero_workers(
    tmp_path: Path,
) -> None:
    """run_batch rejects fewer than one worker."""
    with pytest.raises(
        ValueError,
        match="workers must be at least 1",
    ):
        run_batch(
            config_path=tmp_path / "config.yaml",
            paths_yaml=None,
            n_runs=1,
            workers=0,
            output_dir=tmp_path / "results",
            label="baseline",
            project_root=tmp_path,
        )