#!/usr/bin/env python3
"""
Compare RTN shipment and landfill usage data between Round 1 and Round 2.

Usage:
    conda activate pv_abm
    python compare_rtn_rounds.py

Outputs one multi-sheet Excel file per comparison variant to:
    /Users/pghosh/SOLAR/RTN_Data/Round_Comparison/
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

R1_DIR: str = "/Users/pghosh/SOLAR/RTN_Data/Round_1"
R2_DIR: str = "/Users/pghosh/SOLAR/RTN_Data/Round_2"
OUT_DIR: str = "/Users/pghosh/SOLAR/RTN_Data/Round_Comparison"

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

JOIN_KEY: list[str] = ["Site", "PCA", "State", "Year", "Quarter", "Period"]

LANDFILL_COST_COLS: list[str] = [
    "TotalCost_$",
    "TransportCost_$",
    "LandfillFee_$/kg",
    "DisposalCost_$",
    "Shipped_kg",
    "Distance_km",
]
RECYCLE_COST_COLS: list[str] = [
    "TotalCost_$",
    "TransportCost_$",
    "RecyclingCost_$",
    "Shipped_kg",
    "Distance_km",
]

# Differences smaller than this are treated as noise (dollar-level precision)
EPSILON: float = 1e-2

# ---------------------------------------------------------------------------
# Comparison configuration
# ---------------------------------------------------------------------------

SHIPMENT_PAIRS: list[dict] = [
    {
        "label": "landfill_alllandfills",
        "r1_file": "shipments_landfill_alllandfills[83].csv",
        "r2_file": "shipments_landfill_alllandfills.csv",
        "dest_col": "Landfill",
        "cost_cols": LANDFILL_COST_COLS,
    },
    {
        "label": "landfill_truelandfills",
        "r1_file": "shipments_landfill_truelandfills[40].csv",
        "r2_file": "shipments_landfill_truelandfills.csv",
        "dest_col": "Landfill",
        "cost_cols": LANDFILL_COST_COLS,
    },
    {
        "label": "recycle_alllandfills",
        "r1_file": "shipments_recycle_alllandfills[66].csv",
        "r2_file": "shipments_recycle_alllandfills.csv",
        "dest_col": "Recycler",
        "cost_cols": RECYCLE_COST_COLS,
    },
    {
        "label": "recycle_truelandfills",
        "r1_file": "shipments_recycle_truelandfills[52].csv",
        "r2_file": "shipments_recycle_truelandfills.csv",
        "dest_col": "Recycler",
        "cost_cols": RECYCLE_COST_COLS,
    },
]

USAGE_SUMMARY_PAIRS: list[dict] = [
    {
        "label": "usage_alllandfills",
        "r1_file": "landfill_usage_summary_alllandfills.csv",
        "r2_file": "landfill_usage_summary_alllandfills.csv",
        "join_col": "Landfill",
        "numeric_cols": [
            "Used_kg_total",
            "Capacity_kg",
            "Utilization_%",
            "LandfillFee_$/kg",
            "LandfillFee_$/metric_ton",
        ],
    },
    {
        "label": "usage_truelandfills",
        "r1_file": "landfill_usage_summary_truelandfills.csv",
        "r2_file": "landfill_usage_summary_truelandfills.csv",
        "join_col": "Landfill",
        "numeric_cols": [
            "Used_kg_total",
            "Capacity_kg",
            "Utilization_%",
            "LandfillFee_$/kg",
            "LandfillFee_$/metric_ton",
        ],
    },
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file and strip leading/trailing whitespace from all string columns.

    Parameters:
    path (str): Absolute path to the CSV file.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df: pd.DataFrame = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def _build_cost_changes(
    same_dest: pd.DataFrame,
    dest_col: str,
    dest_r1_col: str,
    cost_cols: list[str],
) -> pd.DataFrame:
    """
    Build a DataFrame of rows where numeric cost/quantity columns changed
    between rounds, for rows that share the same destination.

    Parameters:
    same_dest (pd.DataFrame): Merged rows where the destination is identical in both rounds.
    dest_col (str): Original destination column name (e.g. "Landfill").
    dest_r1_col (str): Suffixed column name for R1 destination (e.g. "Landfill_r1").
    cost_cols (list[str]): Numeric columns to compare (excluding Distance_km, which is
        already captured in destination_changes).

    Returns:
    pd.DataFrame: Rows with at least one cost/quantity change beyond EPSILON.
    """
    compare_cols: list[str] = [c for c in cost_cols if c != "Distance_km"]

    key_data: dict[str, list] = {col: same_dest[col].tolist() for col in JOIN_KEY}
    key_data[dest_col] = same_dest[dest_r1_col].tolist()

    numeric_data: dict[str, list] = {}
    has_change_arr: np.ndarray = np.zeros(len(same_dest), dtype=bool)

    for col in compare_cols:
        r1_col: str = f"{col}_r1"
        r2_col: str = f"{col}_r2"
        if r1_col not in same_dest.columns or r2_col not in same_dest.columns:
            continue

        r1_vals: np.ndarray = same_dest[r1_col].values.astype(float)
        r2_vals: np.ndarray = same_dest[r2_col].values.astype(float)
        diff_vals: np.ndarray = r2_vals - r1_vals

        with np.errstate(divide="ignore", invalid="ignore"):
            pct_vals: np.ndarray = np.where(
                np.abs(r1_vals) > EPSILON,
                diff_vals / r1_vals * 100.0,
                np.nan,
            )

        numeric_data[f"{col}_r1"] = r1_vals.tolist()
        numeric_data[f"{col}_r2"] = r2_vals.tolist()
        numeric_data[f"{col}_diff (r2-r1)"] = diff_vals.tolist()
        numeric_data[f"{col}_pct_change"] = pct_vals.tolist()
        has_change_arr |= np.abs(diff_vals) > EPSILON

    all_rows: pd.DataFrame = pd.DataFrame({**key_data, **numeric_data})
    return all_rows[has_change_arr].reset_index(drop=True)


def _compare_shipments(
    df_r1: pd.DataFrame,
    df_r2: pd.DataFrame,
    dest_col: str,
    cost_cols: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Compare two shipment DataFrames (landfill or recycler) between rounds.

    Produces eight comparison result sets:
    - sites_r1_only: sites present in Round 1 but not Round 2 (with State).
    - sites_r2_only: sites present in Round 2 but not Round 1 (with State).
    - periods_r1_only: (Site, Period, ...) rows present in R1 but not R2.
    - periods_r2_only: (Site, Period, ...) rows present in R2 but not R1.
    - destination_changes: rows in both rounds where the destination facility changed.
    - cost_changes: rows in both rounds with the same destination but different costs.
    - state_breakdown: per-state count of sites that stopped and started using this facility type.
    - mass_summary: total and per-state Shipped_kg comparison between rounds.

    Parameters:
    df_r1 (pd.DataFrame): Round 1 shipment data.
    df_r2 (pd.DataFrame): Round 2 shipment data.
    dest_col (str): Destination column name ("Landfill" or "Recycler").
    cost_cols (list[str]): Numeric columns to diff in the cost_changes sheet.

    Returns:
    dict[str, pd.DataFrame]: Keyed by sheet name.
    """
    # -- Site-level presence --
    site_state_r1: pd.Series = (
        df_r1[["Site", "State"]].drop_duplicates().set_index("Site")["State"]
    )
    site_state_r2: pd.Series = (
        df_r2[["Site", "State"]].drop_duplicates().set_index("Site")["State"]
    )
    sites_r1: set[str] = set(site_state_r1.index)
    sites_r2: set[str] = set(site_state_r2.index)

    r1_only_list: list[str] = sorted(sites_r1 - sites_r2)
    r2_only_list: list[str] = sorted(sites_r2 - sites_r1)
    sites_r1_only: pd.DataFrame = pd.DataFrame({
        "Site": r1_only_list,
        "State": [site_state_r1[s] for s in r1_only_list],
    })
    sites_r2_only: pd.DataFrame = pd.DataFrame({
        "Site": r2_only_list,
        "State": [site_state_r2[s] for s in r2_only_list],
    })

    # -- Period-level outer merge --
    r1_cols: list[str] = [c for c in JOIN_KEY + [dest_col] + cost_cols if c in df_r1.columns]
    r2_cols: list[str] = [c for c in JOIN_KEY + [dest_col] + cost_cols if c in df_r2.columns]

    merged: pd.DataFrame = pd.merge(
        df_r1[r1_cols],
        df_r2[r2_cols],
        on=JOIN_KEY,
        how="outer",
        suffixes=("_r1", "_r2"),
        indicator=True,
    )

    periods_r1_only: pd.DataFrame = (
        merged.loc[merged["_merge"] == "left_only", JOIN_KEY].reset_index(drop=True)
    )
    periods_r2_only: pd.DataFrame = (
        merged.loc[merged["_merge"] == "right_only", JOIN_KEY].reset_index(drop=True)
    )
    in_both: pd.DataFrame = merged[merged["_merge"] == "both"].copy()

    # -- Destination changes --
    dest_r1_col: str = f"{dest_col}_r1"
    dest_r2_col: str = f"{dest_col}_r2"

    dest_mask: pd.Series = in_both[dest_r1_col] != in_both[dest_r2_col]
    dest_changed: pd.DataFrame = in_both.loc[dest_mask, JOIN_KEY + [dest_r1_col, dest_r2_col]].copy()

    if "Distance_km_r1" in in_both.columns:
        dest_changed["Distance_km_r1"] = in_both.loc[dest_mask, "Distance_km_r1"].values
        dest_changed["Distance_km_r2"] = in_both.loc[dest_mask, "Distance_km_r2"].values
        dest_changed["Distance_km_diff (r2-r1)"] = (
            in_both.loc[dest_mask, "Distance_km_r2"].values
            - in_both.loc[dest_mask, "Distance_km_r1"].values
        )

    destination_changes: pd.DataFrame = dest_changed.rename(
        columns={dest_r1_col: f"{dest_col}_Round1", dest_r2_col: f"{dest_col}_Round2"}
    ).reset_index(drop=True)

    # -- Cost changes (same destination) --
    same_dest: pd.DataFrame = in_both[in_both[dest_r1_col] == in_both[dest_r2_col]].copy()
    cost_changes: pd.DataFrame = _build_cost_changes(
        same_dest, dest_col, dest_r1_col, cost_cols
    )

    # -- State breakdown: sites stopped vs. started per state --
    stopped_by_state: pd.Series = (
        sites_r1_only.groupby("State").size().rename("Sites_stopped")
    )
    started_by_state: pd.Series = (
        sites_r2_only.groupby("State").size().rename("Sites_started")
    )
    state_breakdown: pd.DataFrame = (
        pd.concat([stopped_by_state, started_by_state], axis=1)
        .fillna(0)
        .astype(int)
        .assign(Net_change=lambda df: df["Sites_started"] - df["Sites_stopped"])
        .reset_index()
        .sort_values("Sites_stopped", ascending=False)
        .reset_index(drop=True)
    )

    # -- Mass summary: total Shipped_kg by state and overall --
    mass_r1_by_state: pd.Series = df_r1.groupby("State")["Shipped_kg"].sum().rename("Shipped_kg_r1")
    mass_r2_by_state: pd.Series = df_r2.groupby("State")["Shipped_kg"].sum().rename("Shipped_kg_r2")
    mass_by_state: pd.DataFrame = (
        pd.concat([mass_r1_by_state, mass_r2_by_state], axis=1)
        .fillna(0)
        .reset_index()
    )
    mass_by_state["diff_kg (r2-r1)"] = mass_by_state["Shipped_kg_r2"] - mass_by_state["Shipped_kg_r1"]
    with np.errstate(divide="ignore", invalid="ignore"):
        mass_by_state["pct_change"] = np.where(
            mass_by_state["Shipped_kg_r1"] > 0,
            mass_by_state["diff_kg (r2-r1)"] / mass_by_state["Shipped_kg_r1"] * 100.0,
            np.nan,
        )
    total_r1_kg: float = df_r1["Shipped_kg"].sum()
    total_r2_kg: float = df_r2["Shipped_kg"].sum()
    totals_row: pd.DataFrame = pd.DataFrame([{
        "State": "TOTAL",
        "Shipped_kg_r1": total_r1_kg,
        "Shipped_kg_r2": total_r2_kg,
        "diff_kg (r2-r1)": total_r2_kg - total_r1_kg,
        "pct_change": (total_r2_kg - total_r1_kg) / total_r1_kg * 100.0 if total_r1_kg > 0 else np.nan,
    }])
    mass_summary: pd.DataFrame = pd.concat(
        [mass_by_state.sort_values("State"), totals_row], ignore_index=True
    )

    return {
        "sites_r1_only": sites_r1_only,
        "sites_r2_only": sites_r2_only,
        "periods_r1_only": periods_r1_only,
        "periods_r2_only": periods_r2_only,
        "destination_changes": destination_changes,
        "cost_changes": cost_changes,
        "state_breakdown": state_breakdown,
        "mass_summary": mass_summary,
    }


def _compare_usage_summary(
    df_r1: pd.DataFrame,
    df_r2: pd.DataFrame,
    join_col: str,
    numeric_cols: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Compare landfill usage summary DataFrames between rounds.

    Produces four result sets:
    - <join_col>s_r1_only: facilities in Round 1 but not Round 2.
    - <join_col>s_r2_only: facilities in Round 2 but not Round 1.
    - value_changes: facilities in both rounds where any numeric value changed.
    - all_values: all facilities in both rounds with R1/R2/diff columns side by side.

    Parameters:
    df_r1 (pd.DataFrame): Round 1 usage summary.
    df_r2 (pd.DataFrame): Round 2 usage summary.
    join_col (str): Column to join on (e.g. "Landfill").
    numeric_cols (list[str]): Numeric columns to compare.

    Returns:
    dict[str, pd.DataFrame]: Keyed by sheet name.
    """
    entities_r1: set[str] = set(df_r1[join_col].unique())
    entities_r2: set[str] = set(df_r2[join_col].unique())
    only_r1: pd.DataFrame = pd.DataFrame(
        sorted(entities_r1 - entities_r2), columns=[join_col]
    )
    only_r2: pd.DataFrame = pd.DataFrame(
        sorted(entities_r2 - entities_r1), columns=[join_col]
    )

    merged: pd.DataFrame = pd.merge(
        df_r1, df_r2, on=join_col, how="outer", suffixes=("_r1", "_r2"), indicator=True
    )
    in_both: pd.DataFrame = merged[merged["_merge"] == "both"].copy()

    change_data: dict[str, list] = {join_col: in_both[join_col].tolist()}
    has_change_arr: np.ndarray = np.zeros(len(in_both), dtype=bool)

    for col in numeric_cols:
        r1_col: str = f"{col}_r1"
        r2_col: str = f"{col}_r2"
        if r1_col not in in_both.columns or r2_col not in in_both.columns:
            continue
        r1_vals: np.ndarray = in_both[r1_col].values.astype(float)
        r2_vals: np.ndarray = in_both[r2_col].values.astype(float)
        diff_vals: np.ndarray = r2_vals - r1_vals

        change_data[f"{col}_r1"] = r1_vals.tolist()
        change_data[f"{col}_r2"] = r2_vals.tolist()
        change_data[f"{col}_diff (r2-r1)"] = diff_vals.tolist()
        has_change_arr |= np.abs(diff_vals) > EPSILON

    all_values: pd.DataFrame = pd.DataFrame(change_data)
    value_changes: pd.DataFrame = all_values[has_change_arr].reset_index(drop=True)

    return {
        f"{join_col.lower()}s_r1_only": only_r1,
        f"{join_col.lower()}s_r2_only": only_r2,
        "value_changes": value_changes,
        "all_values": all_values,
    }


def _save_excel(results: dict[str, pd.DataFrame], out_path: str) -> None:
    """
    Save a dictionary of DataFrames as named sheets in an Excel file.

    Parameters:
    results (dict[str, pd.DataFrame]): Mapping of sheet name to DataFrame.
    out_path (str): Absolute path for the output .xlsx file.

    Returns:
    None
    """
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name, df in results.items():
            # Excel sheet names are limited to 31 characters
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    logger.info(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Run all Round 1 vs Round 2 comparisons and save results to Excel files
    in OUT_DIR. Skips any pair whose input files cannot be found.

    Returns:
    None
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    # -- Shipment comparisons (4 pairs) --
    for pair in SHIPMENT_PAIRS:
        label: str = pair["label"]
        r1_path: str = os.path.join(R1_DIR, pair["r1_file"])
        r2_path: str = os.path.join(R2_DIR, pair["r2_file"])

        if not os.path.exists(r1_path):
            logger.warning(f"Round 1 file not found, skipping {label}: {r1_path}")
            continue
        if not os.path.exists(r2_path):
            logger.warning(f"Round 2 file not found, skipping {label}: {r2_path}")
            continue

        logger.info(f"Comparing {label} ...")
        df_r1: pd.DataFrame = _load_csv(r1_path)
        df_r2: pd.DataFrame = _load_csv(r2_path)

        results: dict[str, pd.DataFrame] = _compare_shipments(
            df_r1, df_r2, pair["dest_col"], pair["cost_cols"]
        )

        logger.info(
            f"  sites_r1_only={len(results['sites_r1_only']):>5}  "
            f"sites_r2_only={len(results['sites_r2_only']):>5}  "
            f"periods_r1_only={len(results['periods_r1_only']):>6}  "
            f"periods_r2_only={len(results['periods_r2_only']):>6}  "
            f"destination_changes={len(results['destination_changes']):>6}  "
            f"cost_changes={len(results['cost_changes']):>6}"
        )

        out_path: str = os.path.join(OUT_DIR, f"comparison_{label}.xlsx")
        _save_excel(results, out_path)

    # -- Usage summary comparisons (2 pairs) --
    for pair in USAGE_SUMMARY_PAIRS:
        label: str = pair["label"]
        r1_path: str = os.path.join(R1_DIR, pair["r1_file"])
        r2_path: str = os.path.join(R2_DIR, pair["r2_file"])

        if not os.path.exists(r1_path):
            logger.warning(f"Round 1 file not found, skipping {label}: {r1_path}")
            continue
        if not os.path.exists(r2_path):
            logger.warning(f"Round 2 file not found, skipping {label}: {r2_path}")
            continue

        logger.info(f"Comparing usage summary: {label} ...")
        df_r1: pd.DataFrame = _load_csv(r1_path)
        df_r2: pd.DataFrame = _load_csv(r2_path)

        results: dict[str, pd.DataFrame] = _compare_usage_summary(
            df_r1, df_r2, pair["join_col"], pair["numeric_cols"]
        )

        out_path: str = os.path.join(OUT_DIR, f"comparison_{label}.xlsx")
        _save_excel(results, out_path)


if __name__ == "__main__":
    main()
