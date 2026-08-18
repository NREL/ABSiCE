# absice/schemas/data_paths_config.py
"""Pydantic schema for ABSiCE external data paths."""

from pathlib import Path
import yaml
from pydantic import BaseModel, ConfigDict

class DataPathsConfig(BaseModel):
    """
    Paths to all external data files used by the simulation.

    Every field is stored as a Path. File existence is checked later by
    DataLoader so tests can create this configuration without real files.
    """

    model_config = ConfigDict(extra="forbid")

    # PV ICE / ReEDS
    reeds_solar_futures: Path
    reeds_std_scen24: Path
    gis_centroids: Path
    baseline_module_mass: Path
    baseline_module_energy: Path
    pvice_pca_merged_dir: Path
    pvice_pca_dataout_dir: Path

    # Distance matrices
    recycler_distances: Path
    landfill_distances: Path
    hazardous_landfill_distances: Path
    uw_landfill_distances: Path
    uw_recycler_distances: Path

    # Facility data
    recycler_data: Path
    landfill_data: Path
    hazardous_landfill_data: Path
    uw_landfill_data: Path
    uw_recycler_data: Path

    # Market / auxiliary
    correct_mat_factor: Path
    states_adjacency_matrix: Path
    uspvdb: Path

    # Policy
    policy_by_state: Path
    policy_schedule: Path
    tclp_market_share: Path
    generator_threshold: Path
    uw_generator_threshold: Path

    @classmethod
    def from_yaml(cls, path: str | Path, project_root: str | Path | None = None) -> "DataPathsConfig":
        """Load paths from YAML and optionally resolve them from the project root."""
        yaml_path = Path(path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Data paths configuration file not found: {yaml_path}")

        with yaml_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)

        if raw is None:
            raise ValueError(f"Data paths configuration file is empty: {yaml_path}")

        if not isinstance(raw, dict):
            raise ValueError("The top level of data_paths.yaml must be a mapping.")

        if project_root is not None:
            root = Path(project_root).resolve()
            raw = {
                name: path_value if Path(path_value).is_absolute() else root / path_value
                for name, path_value in raw.items()
            }

        return cls.model_validate(raw)

    @classmethod
    def make_default(cls, project_root: str | Path) -> "DataPathsConfig":
        """Construct the default paths relative to the project root."""
        project_root = Path(project_root).resolve()
        temp = project_root / "TEMP"
        pvice = project_root / "PV_ICE"
        sup = pvice / "baselines" / "SupportingMaterial"
        pol = project_root / "policy_regulation"

        return cls(
            reeds_solar_futures=sup / "December Core Scenarios ReEDS Outputs Solar Futures v3a.xlsx",
            reeds_std_scen24=project_root / "ReEDS" / "StdScen24_annual_balancingAreas_Mid_Case_CO2e_95by2035.xlsx",
            gis_centroids=sup / "gis_centroid_n.csv",
            baseline_module_mass=pvice / "baselines" / "baseline_modules_mass_US.csv",
            baseline_module_energy=pvice / "baselines" / "baseline_modules_energy.csv",
            pvice_pca_merged_dir=pvice / "TEMP" / "PCA_merged",
            pvice_pca_dataout_dir=pvice / "TEMP" / "PCA",
            recycler_distances=temp / "site_recycler_distances.csv",
            landfill_distances=temp / "site_landfill_distances.csv",
            hazardous_landfill_distances=temp / "hazardous_site_landfill_distances.csv",
            uw_landfill_distances=temp / "universal_waste_site_landfill_distances.csv",
            uw_recycler_distances=temp / "universal_waste_site_recycler_distances.csv",
            recycler_data=temp / "recycler_data.csv",
            landfill_data=temp / "Landfills_data_2023.csv",
            hazardous_landfill_data=temp / "Landfills_data_SA.csv",
            uw_landfill_data=temp / "Universal_Waste_Landfills_data.csv",
            uw_recycler_data=temp / "Universal_Waste_Recyclers_data.csv",
            correct_mat_factor=temp / "correct_mat_factor.csv",
            states_adjacency_matrix=project_root / "StatesAdjacencyMatrix.csv",
            uspvdb=project_root / "USPVDB" / "uspvdb_v3_0_20250430_with_pca.xlsx",
            policy_by_state=pol / "policy_by_state.csv",
            policy_schedule=pol / "policy_schedule.yaml",
            tclp_market_share=pol / "tclp_market_share_interpolated.csv",
            generator_threshold=pol / "generator_threshold.csv",
            uw_generator_threshold=pol / "universal_waste_generator_threshold.csv",
        )