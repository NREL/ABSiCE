# absice/schemas/simulation_config.py
"""Pydantic schemas for ABSiCE simulation configuration."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
# Import existing enums. Do not redefine them here.
from utils import ConsumerAgentResolution, TIMESTEP

# -----------------------------------------------------------------------------
# Shared configuration behavior
# -----------------------------------------------------------------------------
class ConfigBaseModel(BaseModel):
    """Base class shared by all ABSiCE configuration models."""

    model_config = ConfigDict(
        validate_assignment=True,
        validate_default=True,
        extra="forbid",
    )


# -----------------------------------------------------------------------------
# Enum parsing helpers
# -----------------------------------------------------------------------------
def _parse_timestep(value: object) -> TIMESTEP:
    """
    Convert a YAML string such as ``annual`` into a TIMESTEP enum.

    The existing TIMESTEP enum uses integer values, so Pydantic cannot
    automatically convert the enum member name from YAML.
    """

    if isinstance(value, TIMESTEP):
        return value

    if isinstance(value, str):
        try:
            return TIMESTEP[value.upper()]
        except KeyError as error:
            valid_options = [member.name.lower() for member in TIMESTEP]
            raise ValueError(
                f"Invalid timestep '{value}'. "
                f"Choose from: {valid_options}"
            ) from error

    raise TypeError(
        "Expected timestep to be a string or TIMESTEP value, "
        f"but received {type(value).__name__}."
    )


def _parse_resolution(value: object) -> ConsumerAgentResolution:
    """
    Convert a YAML string such as ``site`` into ConsumerAgentResolution.
    """

    if isinstance(value, ConsumerAgentResolution):
        return value

    if isinstance(value, str):
        try:
            return ConsumerAgentResolution[value.upper()]
        except KeyError as error:
            valid_options = [
                member.name.lower()
                for member in ConsumerAgentResolution
            ]
            raise ValueError(
                f"Invalid consumer resolution '{value}'. "
                f"Choose from: {valid_options}"
            ) from error

    raise TypeError(
        "Expected resolution to be a string or "
        "ConsumerAgentResolution value, "
        f"but received {type(value).__name__}."
    )


# -----------------------------------------------------------------------------
# Default-value helpers
#
# default_factory is used for lists and dictionaries so different configuration
# objects do not accidentally share mutable data.
# -----------------------------------------------------------------------------
def _default_consumer_distribution() -> dict[str, float]:
    return {
        "residential": 1.0, "commercial": 0.0, "utility": 0.0,
    }


def _default_total_number_product() -> list[int]:
    return [
        38, 38, 38, 38, 38, 38, 38, 139, 251, 378, 739, 1670,
        2935, 4146, 5432, 6525, 3609, 4207, 4905, 5719,
    ]


def _default_landfill_cost() -> list[float]:
    return [
        1156, 961, 922, 896, 727, 558, 870, 1429, 1104, 1065,
        1026, 961, 896, 883, 883, 675, 675, 662, 961, 805,
        636, 636, 610, 416, 636, 844, 831, 805, 675, 623,
        623, 571, 545, 506, 506, 584, 714, 649, 636, 571,
        571, 506, 429, 390, 533, 649, 520, 520, 494, 429,
    ]


def _default_hazard_cutoff() -> dict[str, float]:
    state_codes = [
        "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL",
        "GA", "ID", "IL", "IN", "IA", "KS", "KY", "LA",
        "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT",
        "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND",
        "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
        "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    ]

    return {
        "federal": 5.0,
        **{state: 5.0 for state in state_codes},
    }


# -----------------------------------------------------------------------------
# Run and calibration configuration
# -----------------------------------------------------------------------------
class RunConfig(ConfigBaseModel):
    """Core settings controlling one simulation run."""

    seed: int | None = None

    last_step: int = Field(
        default=31, gt=0, description="Number of simulation time steps.",
    )

    timestep: TIMESTEP = Field(
        default=TIMESTEP.ANNUAL, description="Time interval represented by each model step.",
    )

    @field_validator("timestep", mode="before")
    @classmethod
    def coerce_timestep(cls, value: object) -> TIMESTEP:
        """Convert a YAML timestep string into the TIMESTEP enum."""

        return _parse_timestep(value)


class CalibrationConfig(ConfigBaseModel):
    """Calibration and sensitivity-analysis multipliers."""

    calibration_n_sensitivity: float = 1.0
    calibration_n_sensitivity_2: float = 1.0
    calibration_n_sensitivity_3: float = 1.0
    calibration_n_sensitivity_4: float = 1.0
    calibration_n_sensitivity_5: float = 1.0


# -----------------------------------------------------------------------------
# Agent and network configuration
# -----------------------------------------------------------------------------
class NetworkConfig(ConfigBaseModel):
    """Agent counts and network topology settings."""

    num_consumers: int = Field(default=1000, gt=0)
    consumers_node_degree: int = Field(default=10, gt=0)
    consumers_network_type: str = Field(
        default="small-world", min_length=1,
    )
    rewiring_prob: float = Field(
        default=0.1, ge=0.0, le=1.0,
    )

    num_recyclers: int = Field(default=16, gt=0)
    num_producers: int = Field(default=60, gt=0)
    num_refurbishers: int = Field(default=15, gt=0)

    prod_n_recyc_node_degree: int = Field(default=5, gt=0)
    prod_n_recyc_network_type: str = Field(
        default="small-world", min_length=1,
    )


class ConsumerConfig(ConfigBaseModel):
    """Consumer-agent settings and spatial resolution."""

    resolution: ConsumerAgentResolution = (
        ConsumerAgentResolution.SITE
    )

    model_states: list[str] | None = None

    consumers_distribution: dict[str, float] = Field(
        default_factory=_default_consumer_distribution
    )

    product_distribution: dict[str, float] = Field(
        default_factory=_default_consumer_distribution
    )

    @field_validator("resolution", mode="before")
    @classmethod
    def coerce_resolution(
        cls,
        value: object,
    ) -> ConsumerAgentResolution:
        """Convert a YAML resolution string into the existing enum."""

        return _parse_resolution(value)


# -----------------------------------------------------------------------------
# Product and end-of-life configuration
# -----------------------------------------------------------------------------
class ProductConfig(ConfigBaseModel):
    """Product stock, growth, lifetime, and failure parameters."""

    total_number_product: list[int] = Field(
        default_factory=_default_total_number_product
    )

    product_growth: list[float] = Field(
        default_factory=lambda: [0.166, 0.045]
    )

    growth_threshold: int = Field(default=10, gt=0)

    failure_rate_alpha: list[float] = Field(
        default_factory=lambda: [2.4928, 5.3759, 3.93495]
    )

    product_lifetime: int = Field(default=30, gt=0)

    product_average_wght: float = Field(
        default=0.1,
        gt=0.0,
    )

    mass_to_function_reg_coeff: float = Field(
        default=0.03,
        gt=0.0,
    )

    max_storage: list[float] = Field(
        default_factory=lambda: [1.0, 8.0, 4.0]
    )


class EolConfig(ConfigBaseModel):
    """End-of-life pathways and purchase-choice settings."""

    init_eol_rate: dict[str, float] = Field(
        default_factory=lambda: {
            "repair": 0.005,
            "sell": 0.01,
            "recycle": 0.1,
            "landfill": 0.885,
            "hoard": 0.0,
        }
    )

    all_eol_pathways: dict[str, bool] = Field(
        default_factory=lambda: {
            "repair": True,
            "sell": True,
            "recycle": True,
            "landfill": True,
            "hoard": False,
        }
    )

    init_purchase_choice: dict[str, float] = Field(
        default_factory=lambda: {
            "new": 0.9995,
            "used": 0.0005,
            "certified": 0.0,
        }
    )

    purchase_choices: dict[str, bool] = Field(
        default_factory=lambda: {
            "new": True,
            "used": True,
            "certified": False,
        }
    )


# -----------------------------------------------------------------------------
# Theory of Planned Behavior configuration
# -----------------------------------------------------------------------------
class ExtendedTpbConfig(ConfigBaseModel):
    """Optional extended Theory of Planned Behavior settings."""

    enabled: bool = Field(
        default=False,
        alias="Extended tpb",
    )

    w_convenience: float = 0.28
    w_knowledge: float = -0.51

    knowledge_distrib: list[float] = Field(
        default_factory=lambda: [0.5, 0.49]
    )


class TpbConfig(ConfigBaseModel):
    """Theory of Planned Behavior settings and weights."""

    theory_of_planned_behavior: dict[str, bool] = Field(
        default_factory=lambda: {
            "residential": True,
            "commercial": True,
            "utility": True,
        }
    )

    w_sn_eol: float = Field(
        default=0.23,
        ge=0.0,
        le=1.0,
    )
    w_pbc_eol: float = Field(
        default=0.44,
        ge=0.0,
        le=1.0,
    )
    w_a_eol: float = Field(
        default=0.59,
        ge=0.0,
        le=1.0,
    )

    w_sn_reuse: float = Field(
        default=0.497,
        ge=0.0,
        le=1.0,
    )
    w_pbc_reuse: float = Field(
        default=0.382,
        ge=0.0,
        le=1.0,
    )
    w_a_reuse: float = Field(
        default=0.464,
        ge=0.0,
        le=1.0,
    )

    att_distrib_param_eol: list[float] = Field(
        default_factory=lambda: [0.515, 0.1]
    )

    att_distrib_param_reuse: list[float] = Field(
        default_factory=lambda: [0.01, 0.185]
    )

    extended_tpb: ExtendedTpbConfig = Field(
        default_factory=ExtendedTpbConfig
    )


# -----------------------------------------------------------------------------
# Cost configuration
# -----------------------------------------------------------------------------
class CostConfig(ConfigBaseModel):
    """Cost and market-price parameters."""

    hoarding_cost: list[float] = Field(
        default_factory=lambda: [0.0, 130.0, 65.0]
    )

    landfill_cost: list[float] = Field(
        default_factory=_default_landfill_cost
    )

    hazardous_waste_management_cost: dict[str, float] = Field(
        default_factory=lambda: {
            "repair": 0.0,
            "sell": 0.0,
            "recycle": 0.0,
            "landfill": 0.0,
            "hoard": 0.0,
        }
    )

    original_recycling_cost: list[float] = Field(
        default_factory=lambda: [
            400.0 - 1e-6,
            400.0 + 1e-6,
            400.0,
        ]
    )

    recycling_learning_shape_factor: float = -0.01

    repairability: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
    )

    original_repairing_cost: list[float] = Field(
        default_factory=lambda: [
            12987.0,
            58442.0,
            29870.0,
        ]
    )

    repairing_learning_shape_factor: float = -0.31

    scndhand_mkt_pric_rate: list[float] = Field(
        default_factory=lambda: [0.4, 0.2]
    )

    fsthand_mkt_pric: float = 58442.0

    fsthand_mkt_pric_reg_param: list[float] = Field(
        default_factory=lambda: [1.0, 0.04]
    )

    refurbisher_margin: list[float] = Field(
        default_factory=lambda: [0.4, 0.6, 0.5]
    )

    transportation_cost: float = Field(
        default=0.095,
        ge=0.0,
    )

    hazardous_transportation_cost: float = Field(
        default=0.395,
        ge=0.0,
    )

    used_product_substitution_rate: list[float] = Field(
        default_factory=lambda: [0.6, 1.0, 0.8]
    )

    imperfect_substitution: float = Field(
        default=0.0,
        ge=0.0,
    )

    sa_landfill_costs: tuple[bool, float] = (
        False,
        481.0,
    )


# -----------------------------------------------------------------------------
# Material configuration
# -----------------------------------------------------------------------------
class MaterialConfig(ConfigBaseModel):
    """Material composition, prices, waste, and recovery parameters."""

    product_mass_fractions: dict[str, float] = Field(
        default_factory=lambda: {
            "Product": 1.0,
            "Aluminum": 0.08,
            "Glass": 0.76,
            "Copper": 0.01,
            "Insulated cable": 0.012,
            "Silicon": 0.036,
            "Silver": 0.00032,
        }
    )

    established_scd_mkt: dict[str, bool] = Field(
        default_factory=lambda: {
            "Product": True,
            "Aluminum": True,
            "Glass": True,
            "Copper": True,
            "Insulated cable": True,
            "Silicon": False,
            "Silver": False,
        }
    )

    scd_mat_prices: dict[str, list[float]] = Field(
        default_factory=lambda: {
            "Product": [
                float("nan"),
                float("nan"),
                float("nan"),
            ],
            "Aluminum": [0.66, 1.98, 1.32],
            "Glass": [0.01, 0.06, 0.035],
            "Copper": [3.77, 6.75, 5.75],
            "Insulated cable": [3.22, 3.44, 3.33],
            "Silicon": [2.20, 3.18, 2.69],
            "Silver": [453.0, 653.0, 582.0],
        }
    )

    virgin_mat_prices: dict[str, list[float]] = Field(
        default_factory=lambda: {
            "Product": [
                float("nan"),
                float("nan"),
                float("nan"),
            ],
            "Aluminum": [1.76, 2.51, 2.14],
            "Glass": [0.04, 0.07, 0.055],
            "Copper": [4.19, 7.50, 6.39],
            "Insulated cable": [3.22, 3.44, 3.33],
            "Silicon": [2.20, 3.18, 2.69],
            "Silver": [453.0, 653.0, 582.0],
        }
    )

    material_waste_ratio: dict[str, float] = Field(
        default_factory=lambda: {
            "Product": 0.0,
            "Aluminum": 0.0,
            "Glass": 0.0,
            "Copper": 0.0,
            "Insulated cable": 0.0,
            "Silicon": 0.4,
            "Silver": 0.0,
        }
    )

    recovery_fractions: dict[str, float] = Field(
        default_factory=lambda: {
            "Product": float("nan"),
            "Aluminum": 0.92,
            "Glass": 0.85,
            "Copper": 0.72,
            "Insulated cable": 1.0,
            "Silicon": 0.0,
            "Silver": 0.0,
        }
    )


# -----------------------------------------------------------------------------
# Scenario configuration
# -----------------------------------------------------------------------------
class DynamicLifetimeConfig(ConfigBaseModel):
    """Optional dynamic product-lifetime model."""

    enabled: bool = Field(
        default=False,
        alias="Dynamic lifetime",
    )

    d_lifetime_intercept: float = 15.9
    d_lifetime_reg_coeff: float = 0.87

    seed_enabled: bool = Field(
        default=False,
        alias="Seed",
    )

    year: int = Field(
        default=5,
        alias="Year",
        ge=0,
    )

    avg_lifetime: float = Field(
        default=50.0,
        gt=0.0,
    )


class SeedingConfig(ConfigBaseModel):
    """General agent-seeding settings."""

    enabled: bool = Field(
        default=False,
        alias="Seeding",
    )

    year: int = Field(
        default=10,
        alias="Year",
        ge=0,
    )

    number_seed: int = Field(
        default=50,
        ge=0,
    )


class RecyclerSeedingConfig(SeedingConfig):
    """Recycler seeding settings, including the applied discount."""

    discount: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
    )


class RecyclingProcessConfig(ConfigBaseModel):
    """Available recycling-process toggles."""

    frelp: bool = False
    asu: bool = False
    hybrid: bool = False


class ScenarioConfig(ConfigBaseModel):
    """Optional scenario settings and policy-related toggles."""

    recycling_states: list[str] = Field(
        default_factory=lambda: [
            "Texas",
            "Arizona",
            "Oregon",
            "Oklahoma",
            "Wisconsin",
            "Ohio",
            "Kentucky",
            "South Carolina",
        ]
    )

    epr_business_model: bool = False

    recycling_process: RecyclingProcessConfig = Field(
        default_factory=RecyclingProcessConfig
    )

    dynamic_lifetime_model: DynamicLifetimeConfig = Field(
        default_factory=DynamicLifetimeConfig
    )

    seeding: SeedingConfig = Field(
        default_factory=SeedingConfig
    )

    seeding_recyc: RecyclerSeedingConfig = Field(
        default_factory=RecyclerSeedingConfig
    )

    hazardous_waste_regulation_enabled: bool = False

    landfill_solar_waste_acceptance_ratio: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
    )


# -----------------------------------------------------------------------------
# TCLP configuration
# -----------------------------------------------------------------------------
class TclpConfig(ConfigBaseModel):
    """Toxicity Characteristic Leaching Procedure parameters."""

    bsf_mean: float = Field(default=2.18, ge=0.0)
    bsf_std: float = Field(default=0.86, gt=0.0)

    non_bsf_mean: float = Field(default=3.20, ge=0.0)
    non_bsf_std: float = Field(default=1.33, gt=0.0)

    k: float = Field(default=0.30, ge=0.0)
    a50: float = Field(default=15.0, ge=0.0)

    hazard_cutoff: dict[str, float] = Field(
        default_factory=_default_hazard_cutoff
    )

    min_std: float = Field(
        default=0.05,
        gt=0.0,
    )

    distribution: str = Field(
        default="weibull",
        min_length=1,
    )

    @field_validator("distribution")
    @classmethod
    def validate_distribution(cls, value: str) -> str:
        """Allow only distributions currently supported by the model."""

        normalized = value.lower()
        allowed = {"normal", "weibull"}

        if normalized not in allowed:
            raise ValueError(
                f"Invalid TCLP distribution '{value}'. "
                f"Choose from: {sorted(allowed)}"
            )

        return normalized


# -----------------------------------------------------------------------------
# Data-source and calculation flags
# -----------------------------------------------------------------------------
class DataSourceConfig(ConfigBaseModel):
    """Feature flags controlling external datasets and calculations."""

    pv_ice: bool = False
    pca: bool = False
    pca_scenario: bool = False
    geopy: bool = False
    calculate_distances: bool = False
    rtn: bool = False
    solar_cycle: bool = False


# -----------------------------------------------------------------------------
# Legacy file metadata
#
# These settings are included so every current __init__ parameter has a home.
# Actual paths should later move into DataPathsConfig.
# -----------------------------------------------------------------------------
class LandfillDataConfig(ConfigBaseModel):
    """Column names used when reading landfill datasets."""

    landfill_volume_column: str = "$/metric ton"
    landfill_name_column: str = "Facility Name"


class FileNameConfig(ConfigBaseModel):
    """
    Legacy input filenames currently supplied to the model.

    These values should eventually be replaced by DataPathsConfig fields.
    """

    landfill_data: str = Field(
        default="Landfills_data_2023.csv",
        alias="Landfill data",
    )

    pca_landfill_distances: str = Field(
        default="pca_landfills_distances.csv",
        alias="PCA-landfill distances",
    )

    hazardous_landfill_data: str = Field(
        default="Landfills_data_SA.csv",
        alias="Hazardous landfill data",
    )

    hazardous_pca_landfill_distances: str = Field(
        default="pca_landfills_distances_SA.csv",
        alias="Hazardous PCA-landfill distances",
    )

    site_landfill_distances: str = Field(
        default="site_landfills_distances.csv",
        alias="Site-landfill distances",
    )

    recycler_data: str = Field(
        default="Recyclers_data.csv",
        alias="Recycler data",
    )


class LegacyDataConfig(ConfigBaseModel):
    """Temporary home for legacy file and column metadata."""

    landfill_data_params: LandfillDataConfig = Field(
        default_factory=LandfillDataConfig
    )

    file_name: FileNameConfig = Field(
        default_factory=FileNameConfig
    )


# -----------------------------------------------------------------------------
# Top-level configuration
# -----------------------------------------------------------------------------
class SimulationConfig(ConfigBaseModel):
    """
    Top-level configuration for one complete ABSiCE simulation run.

    Each field represents one conceptual area of the simulation.
    """

    run: RunConfig = Field(
        default_factory=RunConfig
    )

    calibration: CalibrationConfig = Field(
        default_factory=CalibrationConfig
    )

    network: NetworkConfig = Field(
        default_factory=NetworkConfig
    )

    consumer: ConsumerConfig = Field(
        default_factory=ConsumerConfig
    )

    product: ProductConfig = Field(
        default_factory=ProductConfig
    )

    eol: EolConfig = Field(
        default_factory=EolConfig
    )

    tpb: TpbConfig = Field(
        default_factory=TpbConfig
    )

    cost: CostConfig = Field(
        default_factory=CostConfig
    )

    material: MaterialConfig = Field(
        default_factory=MaterialConfig
    )

    scenario: ScenarioConfig = Field(
        default_factory=ScenarioConfig
    )

    tclp: TclpConfig = Field(
        default_factory=TclpConfig
    )

    data_source: DataSourceConfig = Field(
        default_factory=DataSourceConfig
    )

    legacy_data: LegacyDataConfig = Field(
        default_factory=LegacyDataConfig
    )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> "SimulationConfig":
        """
        Load and validate configuration from a YAML file.

        Parameters
        ----------
        path
            Path to the YAML configuration file.

        Returns
        -------
        SimulationConfig
            Fully validated simulation configuration.
        """

        yaml_path = Path(path)

        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Simulation configuration file not found: {yaml_path}"
            )

        if not yaml_path.is_file():
            raise ValueError(
                f"Simulation configuration path is not a file: {yaml_path}"
            )

        with yaml_path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file)

        if raw_config is None:
            raise ValueError(
                f"Simulation configuration file is empty: {yaml_path}"
            )

        if not isinstance(raw_config, dict):
            raise ValueError(
                "The top level of the simulation YAML file must be a mapping."
            )

        return cls.model_validate(raw_config)

    def to_yaml(
        self,
        path: str | Path,
    ) -> None:
        """
        Write the current configuration to a YAML file.

        The output uses aliases for fields whose legacy names contain spaces,
        such as ``Dynamic lifetime`` and ``Landfill data``.
        """

        yaml_path = Path(path)
        yaml_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        raw_config = self.model_dump(mode="python", by_alias=True)
        raw_config["run"]["timestep"] = self.run.timestep.name.lower()
        raw_config["consumer"]["resolution"] = self.consumer.resolution.name.lower()

        with yaml_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                raw_config,
                file,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )