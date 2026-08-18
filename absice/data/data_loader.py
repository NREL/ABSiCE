# absice/data/data_loader.py
"""Load all external datasets used by the ABSiCE model."""

import re
from pathlib import Path
import pandas as pd
from enum import Enum
import yaml
from pydantic import BaseModel, ConfigDict
from absice.schemas.data_paths_config import DataPathsConfig

class LoadedData(BaseModel):
    """Typed container holding all datasets needed when the model starts."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    reeds_raw: pd.DataFrame
    gis_centroids: pd.DataFrame
    recycler_data: pd.DataFrame
    landfill_cost_df: pd.DataFrame
    hazardous_landfill_cost_df: pd.DataFrame
    uw_landfill_data: pd.DataFrame
    uw_recycler_data: pd.DataFrame
    recycler_distance_df: pd.DataFrame
    landfill_distance_df: pd.DataFrame
    hazardous_landfill_distance_df: pd.DataFrame
    uw_landfill_distance_df: pd.DataFrame
    uw_recycler_distance_df: pd.DataFrame
    correct_mat_factor: pd.DataFrame
    states_adjacency_matrix: pd.DataFrame
    pvice_waste_eol_df: pd.DataFrame
    tclp_market_share_df: pd.DataFrame
    policy_schedule_by_state: dict[str, dict]
    uspvdb: pd.DataFrame | None
    reeds_data: pd.DataFrame | None

    # Agent-shared data (load once centrally; agents slice/select what they need)
    regulator_policy_by_state: pd.DataFrame
    generator_thresholds: pd.DataFrame
    uw_generator_thresholds: pd.DataFrame
    pca_dataout_by_pca: dict[str, pd.DataFrame]
    pca_datain_by_pca: dict[str, pd.DataFrame]

class DataLoader:
    """Load external files using paths supplied by DataPathsConfig."""

    def __init__(self, paths: DataPathsConfig) -> None:
        self._paths = paths

    def load_all(
        self,
        resolution: str | Enum,
        model_states: list[str] | None = None,
    ) -> LoadedData:
        """Load every dataset and return one LoadedData object."""
        if isinstance(resolution, Enum):
            resolution = resolution.value

        resolution = str(resolution).strip().lower()

        if resolution not in {"site", "pca"}:
            raise ValueError(
                "resolution must be either 'site' or 'pca'; "
                f"received {resolution!r}"
            )

        return LoadedData(
            reeds_raw=self._load_reeds(),
            gis_centroids=self._load_gis(),
            recycler_data=self._load_recycler_data(),
            landfill_cost_df=self._load_landfill_data(),
            hazardous_landfill_cost_df=self._load_hazardous_landfill_data(),
            uw_landfill_data=self._load_csv(self._paths.uw_landfill_data),
            uw_recycler_data=self._load_csv(self._paths.uw_recycler_data),
            recycler_distance_df=self._load_csv(self._paths.recycler_distances),
            landfill_distance_df=self._load_csv(self._paths.landfill_distances),
            hazardous_landfill_distance_df=self._load_csv(self._paths.hazardous_landfill_distances),
            uw_landfill_distance_df=self._load_csv(self._paths.uw_landfill_distances),
            uw_recycler_distance_df=self._load_csv(self._paths.uw_recycler_distances),
            correct_mat_factor=self._load_csv(self._paths.correct_mat_factor),
            states_adjacency_matrix=self._load_csv(self._paths.states_adjacency_matrix),
            pvice_waste_eol_df=self._load_pvice_waste_eol(),
            tclp_market_share_df=self._load_csv(self._paths.tclp_market_share),
            policy_schedule_by_state=self._load_policy_schedule(),
            uspvdb=self._load_uspvdb(model_states) if resolution == "site" else None,
            reeds_data=self._load_reeds_balancing_areas(model_states) if resolution == "site" else None,
            regulator_policy_by_state=self._load_regulator_policy_by_state(),
            generator_thresholds=self._load_generator_thresholds(),
            uw_generator_thresholds=self._load_uw_generator_thresholds(),
            pca_dataout_by_pca=self._load_pca_dataout_by_pca(),
            pca_datain_by_pca=self._load_pca_datain_by_pca(),
        )

    def _load_csv(self, path: Path, **kwargs: object) -> pd.DataFrame:
        """Load a CSV file after confirming that it exists."""
        self._assert_exists(path)
        return pd.read_csv(path, **kwargs)

    def _load_excel(self, path: Path, **kwargs: object) -> pd.DataFrame:
        """Load an Excel file after confirming that it exists."""
        self._assert_exists(path)
        return pd.read_excel(path, **kwargs)

    def _load_reeds(self) -> pd.DataFrame:
        """Load Solar Futures ReEDS installation data."""
        df = self._load_excel(self._paths.reeds_solar_futures, sheet_name="new installs PV")

        if "Tech" in df.columns:
            df = df.drop(columns=["Tech"])

        required_columns = ["Scenario", "Year", "PCA", "State"]
        self._assert_columns(df, required_columns, self._paths.reeds_solar_futures)
        return df.set_index(required_columns)

    def _load_gis(self) -> pd.DataFrame:
        """Load GIS centroid data and use id as the index."""
        df = self._load_csv(self._paths.gis_centroids)
        self._assert_columns(df, ["id"], self._paths.gis_centroids)
        return df.set_index("id")

    def _load_recycler_data(self) -> pd.DataFrame:
        """Load recycler facility data."""
        return self._load_csv(self._paths.recycler_data)

    def _load_landfill_data(self) -> pd.DataFrame:
        """Load landfill facility and cost data."""
        return self._load_csv(self._paths.landfill_data)

    def _load_hazardous_landfill_data(self) -> pd.DataFrame:
        """Load hazardous-waste landfill data."""
        return self._load_csv(self._paths.hazardous_landfill_data)

    def _load_pvice_waste_eol(self) -> pd.DataFrame:
        """Load PV ICE end-of-life waste output."""
        path = self._paths.pvice_pca_merged_dir / "PVICE_PCA_WasteEOL_by_Year_and_PCA.csv"
        return self._load_csv(path)

    def _load_uspvdb(self, model_states: list[str] | None) -> pd.DataFrame:
        """Load USPVDB data and optionally filter by state."""
        df = self._load_excel(self._paths.uspvdb)

        if model_states is not None:
            self._assert_columns(df, ["p_state"], self._paths.uspvdb)
            df = df[df["p_state"].isin(model_states)].copy()

        return df

    def _load_reeds_balancing_areas(self, model_states: list[str] | None) -> pd.DataFrame:
        """Load ReEDS balancing-area data and calculate the PV contribution factor."""
        df = self._load_excel(self._paths.reeds_std_scen24)

        if model_states is not None:
            self._assert_columns(df, ["state"], self._paths.reeds_std_scen24)
            df = df[df["state"].isin(model_states)].copy()

        required_columns = ["upv_MW", "distpv_MW"]
        self._assert_columns(df, required_columns, self._paths.reeds_std_scen24)

        denominator = df["upv_MW"] + df["distpv_MW"]
        df["utility_scale_pv_contribution_factor"] = df["upv_MW"].div(denominator).fillna(0.0)
        return df

    def _load_policy_schedule(self) -> dict[str, dict]:
        """Load policy schedules and reorganize them by state."""
        path = self._paths.policy_schedule

        if not path.exists():
            return {}

        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        raw_policies = config.get("policies") or {}
        schedule_by_state: dict[str, dict] = {}

        for policy_name, entries in raw_policies.items():
            if not entries:
                continue

            for entry in entries:
                states = entry.get("states", [])
                entry_schedule = {key: value for key, value in entry.items() if key != "states"}

                for state in states:
                    schedule_by_state.setdefault(state, {})[policy_name] = entry_schedule

        return schedule_by_state

    def _load_regulator_policy_by_state(self) -> pd.DataFrame:
        """Load the full policy-by-state table used by RegulatorAgents."""
        return self._load_csv(self._paths.policy_by_state)

    def _load_generator_thresholds(self) -> pd.DataFrame:
        """Load the generator threshold table used by RegulatorAgents."""
        return self._load_csv(self._paths.generator_threshold)

    def _load_uw_generator_thresholds(self) -> pd.DataFrame:
        """Load the universal-waste generator threshold table used by RegulatorAgents."""
        return self._load_csv(self._paths.uw_generator_threshold)

    def _load_pca_dataout_by_pca(self) -> dict[str, pd.DataFrame]:
        """
        Load per-PCA dataOut files into a dict keyed by PCA id.

        The DataLoader has no PCA list at load time, so the set of PCAs is
        discovered by globbing pvice_pca_dataout_dir for filenames matching
        dataOut_95-by-35.Adv_{pca}_.csv. Returns {} if the directory is
        absent (mirrors the guard in _load_policy_schedule).
        """
        directory = self._paths.pvice_pca_dataout_dir

        if not directory.exists():
            return {}

        pattern = re.compile(r"^dataOut_95-by-35\.Adv_(.+)_\.csv$")
        result: dict[str, pd.DataFrame] = {}

        for filepath in sorted(directory.glob("dataOut_95-by-35.Adv_*_.csv")):
            match = pattern.match(filepath.name)

            if not match:
                continue

            pca = match.group(1)
            result[pca] = pd.read_csv(filepath)

        return result

    def _load_pca_datain_by_pca(self) -> dict[str, pd.DataFrame]:
        """
        Load per-PCA datain files into a dict keyed by PCA id.

        Mirrors _load_pca_dataout_by_pca: the PCA set is discovered by
        globbing pvice_pca_merged_dir for filenames matching
        datain_95-by-35.Adv_{pca}_.csv. Returns {} if the directory is
        absent (mirrors the guard in _load_policy_schedule).
        """
        directory = self._paths.pvice_pca_merged_dir

        if not directory.exists():
            return {}

        pattern = re.compile(r"^datain_95-by-35\.Adv_(.+)_\.csv$")
        result: dict[str, pd.DataFrame] = {}

        for filepath in sorted(directory.glob("datain_95-by-35.Adv_*_.csv")):
            match = pattern.match(filepath.name)

            if not match:
                continue

            pca = match.group(1)
            result[pca] = pd.read_csv(filepath)

        return result

    @staticmethod
    def _assert_exists(path: Path) -> None:
        """Raise a clear error when a required file is missing."""
        if not path.exists():
            raise FileNotFoundError(
                f"Required data file not found: {path}\n"
                "Check config/data_paths.yaml or DataPathsConfig."
            )

    @staticmethod
    def _assert_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
        """Raise a clear error when a dataset is missing required columns."""
        missing = [column for column in required if column not in df.columns]

        if missing:
            raise ValueError(f"Dataset {path} is missing required columns: {missing}")