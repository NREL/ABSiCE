# absice/data/data_loader.py
"""Load all external datasets used by the ABSiCE model."""

from pathlib import Path
import pandas as pd
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

class DataLoader:
    """Load external files using paths supplied by DataPathsConfig."""

    def __init__(self, paths: DataPathsConfig) -> None:
        self._paths = paths

    def load_all(self, resolution: str, model_states: list[str] | None = None) -> LoadedData:
        """Load every dataset and return one LoadedData object."""
        resolution = resolution.lower()

        if resolution not in {"site", "pca"}:
            raise ValueError("resolution must be either 'site' or 'pca'")

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