# -*- coding: utf-8 -*-
"""
transform_pvice_waste_by_pca.py

Reads PVICE PCA-level waste output and reshapes it into a long-format CSV
with columns: Year, PCA, Total_Waste_EOL_Ton.

Only the pXX_Module columns are used; all other material columns are ignored.
"""

import re
import pandas as pd

INPUT_FILE: str = (
    "/Users/pghosh/SOLAR/"
    "PVICE_PCA_Si_WasteEOL_Method2_Hybrid_ALLDEGBINS_ByPCAMaterial.csv"
)
OUTPUT_FILE: str = (
    "/Users/pghosh/SOLAR/ABSiCE/"
    "PVICE_PCA_WasteEOL_by_Year_and_PCA.csv"
)

_MODULE_COL_PATTERN: re.Pattern = re.compile(r"^(p\d+)_Module$")


def extract_pca_id(column_name: str) -> str:
    """
    Extract the PCA identifier from a column name of the form 'pXX_Module'.

    Parameters:
    column_name (str): Column name in the source CSV.

    Returns:
    str: Lowercase PCA identifier, e.g. 'p100'.
    """
    match: re.Match | None = _MODULE_COL_PATTERN.match(column_name)
    if match is None:
        raise ValueError(f"Column '{column_name}' does not match pXX_Module pattern.")
    return match.group(1)


def transform_waste_csv(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Read the wide-format PVICE waste CSV and reshape it into long format.

    Parameters:
    input_path (str): Path to the source CSV file.
    output_path (str): Path where the output CSV will be written.

    Returns:
    pd.DataFrame: The reshaped long-format dataframe.
    """
    df_wide: pd.DataFrame = pd.read_csv(input_path)

    module_columns: list[str] = [
        col for col in df_wide.columns if _MODULE_COL_PATTERN.match(col)
    ]

    df_module: pd.DataFrame = df_wide[["year"] + module_columns].copy()

    df_long: pd.DataFrame = df_module.melt(
        id_vars="year",
        value_vars=module_columns,
        var_name="pca_column",
        value_name="Yearly_Waste_EOL_Ton",
    )

    df_long["PCA"] = df_long["pca_column"].apply(extract_pca_id)
    df_long = df_long.rename(columns={"year": "Year"})
    df_long = df_long[["year", "pca", "Yearly_Waste_EOL_Ton"]]
    df_long = df_long.sort_values(by=["year", "pca"]).reset_index(drop=True)

    df_long.to_csv(output_path, index=False)
    print(f"Saved {len(df_long):,} rows to {output_path}")

    return df_long


if __name__ == "__main__":
    transform_waste_csv(INPUT_FILE, OUTPUT_FILE)
