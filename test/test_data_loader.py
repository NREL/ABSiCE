# -*- coding:utf-8 -*-
"""
Unit tests for the ABSiCE DataLoader's Phase-1 additions (issue #35).

Covers the private per-PCA dict loaders (_load_pca_dataout_by_pca /
_load_pca_datain_by_pca) and the three regulator-policy loaders. Individual
private loader methods are tested directly against a DataPathsConfig built
from tmp_path so no real data files are required.
"""

from pathlib import Path

import pandas as pd
import pytest

from absice.data.data_loader import DataLoader
from absice.schemas.data_paths_config import DataPathsConfig


def _make_paths(tmp_path: Path, **overrides: Path) -> DataPathsConfig:
    """
    Build a DataPathsConfig with dummy paths for every field.

    DataPathsConfig uses extra="forbid" and has no optional fields, so every
    field must be supplied even when a test only exercises one or two of
    them. Callers override just the fields relevant to what they're testing.
    """
    dummy = tmp_path / "unused.csv"

    fields: dict[str, Path] = {
        "reeds_solar_futures": dummy,
        "reeds_std_scen24": dummy,
        "gis_centroids": dummy,
        "baseline_module_mass": dummy,
        "baseline_module_energy": dummy,
        "pvice_pca_merged_dir": tmp_path / "PCA_merged",
        "pvice_pca_dataout_dir": tmp_path / "PCA",
        "recycler_distances": dummy,
        "landfill_distances": dummy,
        "hazardous_landfill_distances": dummy,
        "uw_landfill_distances": dummy,
        "uw_recycler_distances": dummy,
        "recycler_data": dummy,
        "landfill_data": dummy,
        "hazardous_landfill_data": dummy,
        "uw_landfill_data": dummy,
        "uw_recycler_data": dummy,
        "correct_mat_factor": dummy,
        "states_adjacency_matrix": dummy,
        "uspvdb": dummy,
        "policy_by_state": dummy,
        "policy_schedule": dummy,
        "tclp_market_share": dummy,
        "generator_threshold": dummy,
        "uw_generator_threshold": dummy,
    }
    fields.update(overrides)

    return DataPathsConfig(**fields)


# ---------------------------------------------------------------------------
# _load_pca_dataout_by_pca
# ---------------------------------------------------------------------------

def test_load_pca_dataout_by_pca_reads_matching_files(tmp_path: Path) -> None:
    """Files matching dataOut_95-by-35.Adv_{pca}_.csv are keyed by PCA id."""
    dataout_dir = tmp_path / "PCA"
    dataout_dir.mkdir()

    df_p1 = pd.DataFrame({"year": [2019, 2020], "Yearly_Sum_Power_atEOL": [1.0, 2.0]})
    df_p22 = pd.DataFrame({"year": [2019, 2020], "Yearly_Sum_Power_atEOL": [3.0, 4.0]})
    df_p1.to_csv(dataout_dir / "dataOut_95-by-35.Adv_p1_.csv", index=False)
    df_p22.to_csv(dataout_dir / "dataOut_95-by-35.Adv_p22_.csv", index=False)

    # Decoy file: missing the trailing underscore before .csv, so it must
    # not be picked up by the glob pattern at all.
    (dataout_dir / "dataOut_95-by-35.Adv_p1.csv").write_text("year\n2019\n")

    paths = _make_paths(tmp_path, pvice_pca_dataout_dir=dataout_dir)
    loader = DataLoader(paths)

    result = loader._load_pca_dataout_by_pca()

    assert set(result.keys()) == {"p1", "p22"}
    pd.testing.assert_frame_equal(result["p1"], df_p1)
    pd.testing.assert_frame_equal(result["p22"], df_p22)


def test_load_pca_dataout_by_pca_missing_dir_returns_empty(tmp_path: Path) -> None:
    """A missing pvice_pca_dataout_dir returns {} instead of raising."""
    paths = _make_paths(
        tmp_path,
        pvice_pca_dataout_dir=tmp_path / "does_not_exist",
    )
    loader = DataLoader(paths)

    assert loader._load_pca_dataout_by_pca() == {}


def test_load_pca_dataout_by_pca_ignores_non_matching_filenames(tmp_path: Path) -> None:
    """Files that don't match the dataOut naming pattern are ignored."""
    dataout_dir = tmp_path / "PCA"
    dataout_dir.mkdir()

    df_p1 = pd.DataFrame({"year": [2019], "Yearly_Sum_Power_atEOL": [1.0]})
    df_p1.to_csv(dataout_dir / "dataOut_95-by-35.Adv_p1_.csv", index=False)

    # Unrelated file living in the same directory.
    (dataout_dir / "readme.txt").write_text("not a pca file")
    # A datain-style file should not show up in the dataout results.
    (dataout_dir / "datain_95-by-35.Adv_p1_.csv").write_text("year\n2019\n")

    paths = _make_paths(tmp_path, pvice_pca_dataout_dir=dataout_dir)
    loader = DataLoader(paths)

    result = loader._load_pca_dataout_by_pca()

    assert set(result.keys()) == {"p1"}


# ---------------------------------------------------------------------------
# _load_pca_datain_by_pca
# ---------------------------------------------------------------------------

def test_load_pca_datain_by_pca_reads_matching_files(tmp_path: Path) -> None:
    """Files matching datain_95-by-35.Adv_{pca}_.csv are keyed by PCA id."""
    datain_dir = tmp_path / "PCA_merged"
    datain_dir.mkdir()

    df_p5 = pd.DataFrame(
        {"year": [2019, 2020], "new_Installed_Capacity_[MW]": [10.0, 20.0]}
    )
    df_p7 = pd.DataFrame(
        {"year": [2019, 2020], "new_Installed_Capacity_[MW]": [30.0, 40.0]}
    )
    df_p5.to_csv(datain_dir / "datain_95-by-35.Adv_p5_.csv", index=False)
    df_p7.to_csv(datain_dir / "datain_95-by-35.Adv_p7_.csv", index=False)

    paths = _make_paths(tmp_path, pvice_pca_merged_dir=datain_dir)
    loader = DataLoader(paths)

    result = loader._load_pca_datain_by_pca()

    assert set(result.keys()) == {"p5", "p7"}
    pd.testing.assert_frame_equal(result["p5"], df_p5)
    pd.testing.assert_frame_equal(result["p7"], df_p7)


def test_load_pca_datain_by_pca_missing_dir_returns_empty(tmp_path: Path) -> None:
    """A missing pvice_pca_merged_dir returns {} instead of raising."""
    paths = _make_paths(
        tmp_path,
        pvice_pca_merged_dir=tmp_path / "does_not_exist",
    )
    loader = DataLoader(paths)

    assert loader._load_pca_datain_by_pca() == {}


def test_load_pca_datain_by_pca_ignores_non_matching_filenames(tmp_path: Path) -> None:
    """Files that don't match the datain naming pattern are ignored."""
    datain_dir = tmp_path / "PCA_merged"
    datain_dir.mkdir()

    df_p5 = pd.DataFrame(
        {"year": [2019], "new_Installed_Capacity_[MW]": [10.0]}
    )
    df_p5.to_csv(datain_dir / "datain_95-by-35.Adv_p5_.csv", index=False)

    # Missing trailing underscore before .csv: must not be picked up.
    (datain_dir / "datain_95-by-35.Adv_p5.csv").write_text("year\n2019\n")
    (datain_dir / "PVICE_PCA_WasteEOL_by_Year_and_PCA.csv").write_text("year\n2019\n")

    paths = _make_paths(tmp_path, pvice_pca_merged_dir=datain_dir)
    loader = DataLoader(paths)

    result = loader._load_pca_datain_by_pca()

    assert set(result.keys()) == {"p5"}


# ---------------------------------------------------------------------------
# Regulator policy loaders
# ---------------------------------------------------------------------------

def test_load_regulator_policy_by_state_reads_csv(tmp_path: Path) -> None:
    """_load_regulator_policy_by_state reads the full policy_by_state table."""
    policy_path = tmp_path / "policy_by_state.csv"
    expected = pd.DataFrame(
        {
            "state": ["CA", "WA"],
            "universal_waste_regulation": [True, False],
        }
    )
    expected.to_csv(policy_path, index=False)

    paths = _make_paths(tmp_path, policy_by_state=policy_path)
    loader = DataLoader(paths)

    result = loader._load_regulator_policy_by_state()

    pd.testing.assert_frame_equal(result, expected)


def test_load_generator_thresholds_reads_csv(tmp_path: Path) -> None:
    """_load_generator_thresholds reads the full generator_threshold table."""
    threshold_path = tmp_path / "generator_threshold.csv"
    expected = pd.DataFrame(
        {
            "state": ["FED", "CA"],
            "generator_size": ["very_small", "small"],
            "max_storage_kg": [100, 1000],
        }
    )
    expected.to_csv(threshold_path, index=False)

    paths = _make_paths(tmp_path, generator_threshold=threshold_path)
    loader = DataLoader(paths)

    result = loader._load_generator_thresholds()

    pd.testing.assert_frame_equal(result, expected)


def test_load_uw_generator_thresholds_reads_csv(tmp_path: Path) -> None:
    """_load_uw_generator_thresholds reads the universal-waste threshold table."""
    threshold_path = tmp_path / "universal_waste_generator_threshold.csv"
    expected = pd.DataFrame(
        {
            "state": ["FED"],
            "generator_size": ["very_small"],
            "waste_generation_limit_kg": [50],
        }
    )
    expected.to_csv(threshold_path, index=False)

    paths = _make_paths(tmp_path, uw_generator_threshold=threshold_path)
    loader = DataLoader(paths)

    result = loader._load_uw_generator_thresholds()

    pd.testing.assert_frame_equal(result, expected)


def test_load_regulator_policy_by_state_missing_file_raises(tmp_path: Path) -> None:
    """Unlike the per-PCA dict loaders, the policy loaders assert existence."""
    paths = _make_paths(
        tmp_path,
        policy_by_state=tmp_path / "does_not_exist.csv",
    )
    loader = DataLoader(paths)

    with pytest.raises(FileNotFoundError):
        loader._load_regulator_policy_by_state()
