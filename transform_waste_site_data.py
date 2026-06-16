import argparse
import datetime
import os

import pandas as pd


_WASTE_COLUMNS: list[str] = [
    "Waste Repair (Kg)",
    "Waste Sell (Kg)",
    "Waste Recycle (Kg)",
    "Waste Landfill (Kg)",
    "Waste Hoard (Kg)",
]

_ID_COLUMNS: list[str] = [
    "Year",
    "Quarter",
    "PCA",
    "State",
    "Name",
    "Latitude",
    "Longitude",
]

_OUT_TOTAL_MASS: str = "Total mass by site (kg), cumulative"
_OUT_RECYCLED_MASS: str = "Total recycled mass by site (kg), cumulative"
_OUT_RECYCLING_RATE: str = "Recycling rate (%), from cumulative mass values in site"


def _transform_to_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a source waste-by-site DataFrame into the format defined by the
    reference file.  Retains the seven identifier columns and computes three
    summary columns; all other source columns are dropped.

    Parameters:
    df (pd.DataFrame): Source DataFrame containing the waste columns and the
        seven identifier columns.

    Returns:
    pd.DataFrame: Transformed DataFrame with identifier columns plus the three
        summary columns.
    """
    out: pd.DataFrame = df[_ID_COLUMNS].copy()

    total_mass_kg: pd.Series = df[_WASTE_COLUMNS].sum(axis=1)
    recycled_mass_kg: pd.Series = df["Waste Recycle (Kg)"]

    out[_OUT_TOTAL_MASS] = total_mass_kg
    out[_OUT_RECYCLED_MASS] = recycled_mass_kg
    out[_OUT_RECYCLING_RATE] = (recycled_mass_kg / total_mass_kg * 100).fillna(0)

    return out


def transform_waste_site_data(
    source_all_landfills: str,
    source_true_landfills: str,
    output_dir: str,
) -> None:
    """
    Read both source waste-by-site CSV files, transform them into the reference
    format, and write the results to two dated output files.

    Parameters:
    source_all_landfills (str): Path to the all-landfills source CSV file.
    source_true_landfills (str): Path to the true-landfills source CSV file.
    output_dir (str): Directory where the two output CSV files will be written.

    Returns:
    None
    """
    date_prefix: str = datetime.date.today().strftime("%m%d%y")

    df_all: pd.DataFrame = pd.read_csv(source_all_landfills)
    df_true: pd.DataFrame = pd.read_csv(source_true_landfills)

    out_all: pd.DataFrame = _transform_to_format(df_all)
    out_true: pd.DataFrame = _transform_to_format(df_true)

    os.makedirs(output_dir, exist_ok=True)

    path_all: str = os.path.join(
        output_dir,
        f"{date_prefix}_waste_kg_per_year_by_site_all_landfills.csv",
    )
    path_true: str = os.path.join(
        output_dir,
        f"{date_prefix}_waste_kg_per_year_by_site_true_landfills.csv",
    )

    out_all.to_csv(path_all, index=False)
    out_true.to_csv(path_true, index=False)

    print(f"Wrote {len(out_all)} rows to {path_all}")
    print(f"Wrote {len(out_true)} rows to {path_true}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Transform waste-by-site CSVs into the reference format and write "
            "dated output files."
        )
    )
    parser.add_argument(
        "--source_all_landfills",
        type=str,
        default="results/RTN_run_all_landfills/waste_kg_per_year_by_site.csv",
        help="Path to the all-landfills source CSV file.",
    )
    parser.add_argument(
        "--source_true_landfills",
        type=str,
        default=(
            "results/RTN_run_true_landfills/waste_kg_per_year_by_site.csv"
        ),
        help="Path to the true-landfills source CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory where output files will be written (default: 'results').",
    )

    args = parser.parse_args()

    transform_waste_site_data(
        source_all_landfills=args.source_all_landfills,
        source_true_landfills=args.source_true_landfills,
        output_dir=args.output_dir,
    )
