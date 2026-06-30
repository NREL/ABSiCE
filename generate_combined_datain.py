# -*- coding: utf-8 -*-
"""
generate_combined_datain.py

Creates per-PCA datain CSV files with updated installed capacity values
by combining:
  - Historical data (2010-2025): from existing PV ICE datain CSVs
  - Future data (2026+): from the ReEDS scenario Excel file

ReEDS reports cumulative capacity at 3-year intervals (2026, 2029, 2032, ...).
Each 3-year value (upv_MW + distpv_MW) is redistributed evenly across the
corresponding 3-year window (÷3 per year) to produce annual additions.

Output: one updated CSV per PCA in OUTPUT_DIR, retaining all non-capacity
columns from the original datain files unchanged.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Directory containing the original per-PCA datain CSVs
DATAIN_DIR: str = os.path.join(
    os.path.dirname(__file__), "PV_ICE", "TEMP", "PCA"
)

# ReEDS Excel file (new scenario starting 2026, 3-year intervals)
REEDS_FILE: str = os.path.join(
    os.path.dirname(__file__),
    "ReEDS",
    "StdScen24_annual_balancingAreas_Mid_Case_CO2e_95by2035.xlsx",
)

# Output directory for the merged datain files
OUTPUT_DIR: str = os.path.join(
    os.path.dirname(__file__), "PV_ICE", "TEMP", "PCA_merged"
)

# Year at which ReEDS data takes over from the historical datain files
REEDS_START_YEAR: int = 2026

# Interval of the ReEDS data (years between consecutive ReEDS data points)
REEDS_INTERVAL: int = 3

# Capacity column in the datain CSV
CAPACITY_COLUMN: str = "new_Installed_Capacity_[MW]"


def load_reeds_data(reeds_file: str) -> pd.DataFrame:
    """
    Load the ReEDS Excel file and compute annual installed capacity per PCA.

    The ReEDS file reports total accumulated capacity at 3-year intervals.
    This function converts those to annual additions by dividing the combined
    (upv_MW + distpv_MW) value by the 3-year interval.

    Parameters:
        reeds_file (str): Path to the ReEDS Excel file.

    Returns:
        pd.DataFrame: DataFrame with columns ['r', 'year',
            'new_Installed_Capacity_[MW]'] where each row is a single
            (PCA, year) combination with the annual capacity addition in MW.
    """
    reeds_raw: pd.DataFrame = pd.read_excel(reeds_file)

    # Keep only the columns we need
    reeds_raw = reeds_raw[["r", "t", "upv_MW", "distpv_MW"]].copy()
    reeds_raw = reeds_raw.rename(columns={"r": "pca", "t": "reeds_year"})

    # Total capacity assigned to each 3-year period (MW)
    reeds_raw["total_MW"] = reeds_raw["upv_MW"] + reeds_raw["distpv_MW"]

    # Annual addition = total for the 3-year period ÷ interval.
    # Apply 85% scaling to account for silicon PV share of total capacity,
    # consistent with the assumption used in the original datain file generation.
    reeds_raw["annual_MW"] = reeds_raw["total_MW"] / REEDS_INTERVAL * 0.85

    # Expand each 3-year row into REEDS_INTERVAL annual rows
    rows: list[dict] = []
    for _, row in reeds_raw.iterrows():
        base_year: int = int(row["reeds_year"])
        for offset in range(REEDS_INTERVAL):
            year: int = base_year + offset
            rows.append(
                {
                    "pca": row["pca"],
                    "year": year,
                    CAPACITY_COLUMN: row["annual_MW"],
                }
            )

    reeds_annual: pd.DataFrame = pd.DataFrame(rows)
    return reeds_annual


def build_merged_datain(
    pca: str,
    datain_df: pd.DataFrame,
    reeds_annual: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a merged datain DataFrame for a single PCA by replacing the
    installed capacity values for years >= REEDS_START_YEAR with ReEDS-
    derived annual values.

    Parameters:
        pca (str): PCA identifier (e.g. 'p31').
        datain_df (pd.DataFrame): Original datain CSV loaded as a DataFrame.
        reeds_annual (pd.DataFrame): Annual ReEDS capacity table (all PCAs).

    Returns:
        pd.DataFrame: Merged DataFrame with updated capacity column,
            same columns and row order as the original datain file.
    """
    merged: pd.DataFrame = datain_df.copy()

    # Historical rows: keep as-is (year < REEDS_START_YEAR)
    # Future rows: replace capacity column with ReEDS value

    pca_reeds: pd.DataFrame = reeds_annual[reeds_annual["pca"] == pca].set_index("year")

    for idx, row_data in merged.iterrows():
        year: int = int(row_data["year"])
        if year >= REEDS_START_YEAR and year in pca_reeds.index:
            merged.at[idx, CAPACITY_COLUMN] = pca_reeds.at[year, CAPACITY_COLUMN]

    return merged


def main() -> None:
    """
    Main entry point: load data, merge per PCA, and write output CSVs.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading ReEDS data from:\n  {REEDS_FILE}")
    reeds_annual: pd.DataFrame = load_reeds_data(REEDS_FILE)
    reeds_pcas: set[str] = set(reeds_annual["pca"].unique())
    print(f"  Found {len(reeds_pcas)} PCAs in ReEDS data, "
          f"years {reeds_annual['year'].min()}–{reeds_annual['year'].max()}")

    datain_files: list[str] = [
        f for f in os.listdir(DATAIN_DIR)
        if f.startswith("datain_95-by-35.Adv_") and f.endswith("_.csv")
    ]
    print(f"\nProcessing {len(datain_files)} datain files from:\n  {DATAIN_DIR}")

    matched: int = 0
    unmatched: list[str] = []

    for filename in sorted(datain_files):
        # Extract PCA id from filename, e.g. "datain_95-by-35.Adv_p31_.csv" → "p31"
        pca: str = filename.removeprefix("datain_95-by-35.Adv_").removesuffix("_.csv")

        datain_path: str = os.path.join(DATAIN_DIR, filename)
        datain_df: pd.DataFrame = pd.read_csv(datain_path)

        if pca not in reeds_pcas:
            unmatched.append(pca)
            # Write original file unchanged so the output directory is complete
            out_path: str = os.path.join(OUTPUT_DIR, filename)
            datain_df.to_csv(out_path, index=False)
            continue

        merged_df: pd.DataFrame = build_merged_datain(pca, datain_df, reeds_annual)
        out_path = os.path.join(OUTPUT_DIR, filename)
        merged_df.to_csv(out_path, index=False)
        matched += 1

    print(f"\nDone.")
    print(f"  Updated with ReEDS data : {matched} files")
    print(f"  No ReEDS match (copied unchanged): {len(unmatched)} files")
    if unmatched:
        print(f"  Unmatched PCAs: {', '.join(sorted(unmatched))}")
    print(f"\nOutput written to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
