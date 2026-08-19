# -*- coding: utf-8 -*-
"""
Pure cost-scaling helpers for HPC calibration / recycling-sensitivity runs.

This module computes recycling-cost overrides for one scenario, in two
independent modes:

- RTN mode (``rtn=True``): scales the RTN recycling-shipment cost column by
  a ratio, regenerates the derived recycling-cost CSV under ``RTN/``, and
  returns the resulting filename so it can be assigned to
  ``config.legacy_data.file_name["Recycling data"]``. All filesystem access
  to ``RTN_Data`` is confined to functions in this branch and is only
  exercised when the caller explicitly asks for RTN mode.

- Non-RTN mode (``rtn=False``): returns a scaled copy of
  ``config.cost.original_recycling_cost``. This mode performs no filesystem
  access and has no dependency on ``RTN_Data``.

Transport-cost scaling and the landfill-cost CSV are intentionally out of
scope for this module (transport sensitivity is out of scope this session).

Ported and adapted from (not checked out locally, referenced via
``git show``):
    landfill-paper-june-2026:run_rtn_scenarios.py  (_build_suffix, _LANDFILL_SETS,
        _prepare_cost_files — recycling branch only)
    landfill-paper-june-2026:scale_recycling_cost.py (_scale_file)
and the current-branch ``generate_recycling_costs.py``.
"""

from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------------------
# Baseline constants
# -----------------------------------------------------------------------------

# Baseline recycling rate in $/kg, consistent across all RTN shipment rows.
# RecyclingCost_$ = Shipped_kg x BASELINE_RECYCLING_RATE_PER_KG.
# Used only to convert an absolute cost rate ($/kg) into a scale ratio for
# RTN mode; it has no meaning for the non-RTN triangular-cost lever.
BASELINE_RECYCLING_RATE_PER_KG: float = 0.40

_HPC_DIR: Path = Path(__file__).parent.resolve()
_WORKSPACE_DIR: Path = _HPC_DIR.parent
_RTN_DIR: Path = _WORKSPACE_DIR / "RTN"
_USPVDB_FILE: Path = (
    _WORKSPACE_DIR / "USPVDB" / "uspvdb_v3_0_20250430_with_pca.xlsx"
)

# Candidate locations for the raw RTN shipment data. Only resolved (and only
# required to exist) when RTN mode is actually used.
_RTN_DATA_CANDIDATES: list[Path] = [
    Path("/projects/pvabm/pghosh/RTN_Data/Round_3"),  # HPC (Kestrel)
    Path("/Users/pghosh/SOLAR/RTN_Data/Round_3"),      # macOS local
]

# Recycling-only view of the landfill-set file mapping. Keys select which
# RTN shipment file and derived recycling-cost filename prefix a scenario
# uses. Non-RTN callers may still use these keys purely as a results-label
# prefix; they never touch the shipment_file path in that mode.
LANDFILL_SETS: dict[str, dict[str, str]] = {
    "all_landfills": {
        "shipment_filename": "shipments_recycle_alllandfills.csv",
        "recycling_file_prefix": "RecyclingCostsbyYearAllLandfills",
        "results_prefix": "RTN_run_all_landfills",
        # Unscaled RTN landfill-cost CSV for this set (landfill-cost scaling
        # is out of scope this session; always the base file under RTN/).
        "landfill_filename": "LandfillCostsbyYearAllLandfills.csv",
    },
    "true_landfills": {
        "shipment_filename": "shipments_recycle_truelandfills.csv",
        "recycling_file_prefix": "RecyclingCostsbyYearTrueLandfills",
        "results_prefix": "RTN_run_true_landfills",
        "landfill_filename": "LandfillCostsbyYearTrueLandfills.csv",
    },
}


# -----------------------------------------------------------------------------
# Ratio helpers (mode-agnostic, no filesystem access)
# -----------------------------------------------------------------------------
def ratio_from_cost_rate(
    cost_rate: float,
    baseline_rate: float = BASELINE_RECYCLING_RATE_PER_KG,
) -> float:
    """
    Convert an absolute RTN recycling cost rate ($/kg) into a scale ratio.

    This conversion is only meaningful for RTN mode, where shipment costs
    are computed from a $/kg rate. In non-RTN mode the recycling lever is a
    triangular cost distribution in $/functional-unit, so a $/kg-derived
    ratio must NOT be applied blindly — callers targeting non-RTN mode
    should supply ``ratio`` directly instead of converting a cost rate.

    Parameters
    ----------
    cost_rate
        Absolute recycling cost rate in $/kg (e.g. 0.40 for baseline).
    baseline_rate
        Baseline rate the ratio is computed relative to
        (default: :data:`BASELINE_RECYCLING_RATE_PER_KG`).

    Returns
    -------
    float
        Scale ratio, e.g. ``cost_rate / baseline_rate``.
    """
    return cost_rate / baseline_rate


def build_suffix(ratio: float, cost_component: str = "recycling") -> str:
    """
    Build a filename/label suffix from a scale ratio and cost component.

    Parameters
    ----------
    ratio
        Multiplier (e.g. 1.05 for +5%, 0.9 for -10%).
    cost_component
        ``"recycling"`` (default) or ``"transport"``. Transport scenarios
        get a ``_transport`` infix so they cannot collide with recycling
        scenario labels at the same ratio. Transport is out of scope this
        session but the infix is kept for forward compatibility.

    Returns
    -------
    str
        Suffix string, e.g. ``"_1.05"``, ``"_neg0.9"``,
        ``"_transport_1.05"``, ``"_transport_neg0.9"``.
    """
    infix: str = "_transport" if cost_component == "transport" else ""
    if ratio >= 1.0:
        return f"{infix}_{ratio:g}"
    return f"{infix}_neg{ratio:g}"


def scale_original_recycling_cost(
    original_recycling_cost: list[float],
    ratio: float,
) -> list[float]:
    """
    Scale the non-RTN triangular recycling-cost lever by ``ratio``.

    Performs no filesystem access. Suitable for
    ``config.cost.original_recycling_cost`` (``[left, right, mode]`` used by
    ``np.random.triangular``).

    Parameters
    ----------
    original_recycling_cost
        The baseline ``[left, right, mode]`` triangular cost parameters.
    ratio
        Multiplier applied to every element.

    Returns
    -------
    list[float]
        ``[c * ratio for c in original_recycling_cost]``.
    """
    return [c * ratio for c in original_recycling_cost]


# -----------------------------------------------------------------------------
# RTN-mode filesystem helpers (only exercised when rtn=True)
# -----------------------------------------------------------------------------
def _resolve_rtn_data_dir() -> Path:
    """
    Locate the raw RTN shipment-data directory.

    Returns
    -------
    Path
        The first existing candidate directory.

    Raises
    ------
    FileNotFoundError
        If none of the known candidate directories exist. Only called from
        RTN-mode code paths, so importing this module never requires
        ``RTN_Data`` to be present.
    """
    for candidate in _RTN_DATA_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "RTN mode requires the raw RTN shipment data directory, but none of "
        f"the known locations exist: {_RTN_DATA_CANDIDATES}. "
        "Set up RTN_Data or run with rtn=False."
    )


def _scale_shipment_file(
    file_path: Path,
    scale: float,
    cost_col: str,
    transport_col: str,
    total_col: str,
    file_suffix: str,
    output_dir: Path,
) -> Path:
    """
    Scale one cost column of an RTN shipment CSV and save the result.

    Ported from ``landfill-paper-june-2026:scale_recycling_cost.py``
    (``_scale_file``).

    Parameters
    ----------
    file_path
        Path to the input shipment CSV file.
    scale
        Multiplier applied to ``cost_col``.
    cost_col
        Name of the cost column to scale (e.g. ``"RecyclingCost_$"``).
    transport_col
        Name of the other cost column used to recompute the total.
    total_col
        Name of the total-cost column to recompute.
    file_suffix
        Suffix inserted before the file extension in the output filename.
    output_dir
        Directory to write the scaled CSV into. Deliberately NOT the
        source file's own directory: ``RTN_Data`` may be a read-only,
        externally-managed dataset on the cluster, and writing scaled
        intermediates there also risks a cross-job race when multiple
        scenarios share the same ``RTN_Data`` mount. Use a writable,
        per-project directory instead (e.g. ``RTN/``).

    Returns
    -------
    Path
        Path to the newly written, scaled CSV file, under ``output_dir``.

    Raises
    ------
    ValueError
        If any of ``cost_col``, ``transport_col``, or ``total_col`` is
        missing from the input file.
    """
    df: pd.DataFrame = pd.read_csv(file_path)

    missing: list[str] = [
        col
        for col in (cost_col, transport_col, total_col)
        if col not in df.columns
    ]
    if missing:
        raise ValueError(
            f"{file_path.name}: column(s) not found: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df[cost_col] = df[cost_col] * scale
    df[total_col] = df[transport_col] + df[cost_col]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path: Path = output_dir / (file_path.stem + file_suffix + file_path.suffix)
    df.to_csv(out_path, index=False)
    return out_path


def prepare_rtn_recycling_cost_file(
    landfill_set: str,
    ratio: float,
    suffix: str,
    uspvdb_file: Path | None = None,
    rtn_dir: Path | None = None,
    rtn_data_dir: Path | None = None,
) -> str:
    """
    Scale RTN recycling-shipment costs and regenerate the model-facing CSV.

    For ``ratio == 1.0``, no files are generated; the unscaled base
    recycling-cost CSV already present under ``RTN/`` is used as-is.

    Parameters
    ----------
    landfill_set
        Key into :data:`LANDFILL_SETS` (``"all_landfills"`` or
        ``"true_landfills"``).
    ratio
        Scale multiplier applied to ``RecyclingCost_$``.
    suffix
        Filename suffix, typically produced by :func:`build_suffix`.
    uspvdb_file
        Path to the USPVDB Excel file. Defaults to
        ``USPVDB/uspvdb_v3_0_20250430_with_pca.xlsx`` under the project
        root.
    rtn_dir
        Output directory for the derived recycling-cost CSV. Defaults to
        ``RTN/`` under the project root.
    rtn_data_dir
        Directory containing the raw RTN shipment CSVs. Defaults to the
        first existing entry in the known local/HPC candidate paths.

    Returns
    -------
    str
        Filename (not a full path) of the recycling-cost CSV under
        ``rtn_dir``, suitable for
        ``config.legacy_data.file_name["Recycling data"]``.

    Raises
    ------
    KeyError
        If ``landfill_set`` is not a recognized key.
    FileNotFoundError
        If the RTN shipment data or USPVDB file cannot be located.
    """
    if landfill_set not in LANDFILL_SETS:
        raise KeyError(
            f"Unknown landfill_set '{landfill_set}'. "
            f"Choose from: {list(LANDFILL_SETS.keys())}"
        )

    set_config = LANDFILL_SETS[landfill_set]
    recycling_file_prefix: str = set_config["recycling_file_prefix"]
    base_recycling_filename: str = f"{recycling_file_prefix}.csv"

    rtn_dir = rtn_dir or _RTN_DIR
    rtn_dir.mkdir(parents=True, exist_ok=True)

    if abs(ratio - 1.0) < 1e-9:
        base_path = rtn_dir / base_recycling_filename
        if not base_path.exists():
            raise FileNotFoundError(
                f"Base RTN recycling-cost file not found: {base_path}"
            )
        return base_recycling_filename

    recycling_filename: str = f"{recycling_file_prefix}{suffix}.csv"
    output_path: Path = rtn_dir / recycling_filename
    if output_path.exists():
        return recycling_filename

    rtn_data_dir = rtn_data_dir or _resolve_rtn_data_dir()
    shipment_file: Path = rtn_data_dir / set_config["shipment_filename"]
    if not shipment_file.exists():
        raise FileNotFoundError(
            f"RTN shipment file not found: {shipment_file}"
        )

    uspvdb_file = uspvdb_file or _USPVDB_FILE
    if not Path(uspvdb_file).exists():
        raise FileNotFoundError(f"USPVDB file not found: {uspvdb_file}")

    # Import here (not at module scope) so RTN dependencies are only
    # required when RTN mode is actually exercised.
    from generate_recycling_costs import generate_recycling_costs

    # Write the scaled shipment intermediate under rtn_dir (writable, e.g.
    # RTN/), NOT next to the source file in rtn_data_dir: RTN_Data may be a
    # read-only, externally-managed mount on the cluster, and sharing it
    # across concurrently-running scenario jobs risks a write race (R3).
    scaled_shipment: Path = rtn_dir / (shipment_file.stem + suffix + shipment_file.suffix)
    if scaled_shipment.exists():
        # Idempotent: avoid rescaling a shipment file that a previous run
        # (or a concurrent job for the same scenario) already produced.
        pass
    else:
        scaled_shipment = _scale_shipment_file(
            file_path=shipment_file,
            scale=ratio,
            cost_col="RecyclingCost_$",
            transport_col="TransportCost_$",
            total_col="TotalCost_$",
            file_suffix=suffix,
            output_dir=rtn_dir,
        )

    generate_recycling_costs(
        shipments_file=str(scaled_shipment),
        uspvdb_file=str(uspvdb_file),
        output_file=str(output_path),
    )

    return recycling_filename
