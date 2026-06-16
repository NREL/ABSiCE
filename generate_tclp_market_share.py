"""
Generate tclp_market_share_interpolated.csv from tclp_market_share.csv.

Reads the original TCLP market share data (sparse, irregular year gaps),
linearly interpolates to produce a value for every year between the first
and last year specified, rounds to 2 decimal places, and writes the result
to a new file — leaving the original untouched.
"""

import pandas as pd
import argparse
from pathlib import Path


def generate_tclp_market_share(
    input_file: str,
    output_file: str,
) -> None:
    """
    Linearly interpolate TCLP market share data for every year in range.

    Args:
        input_file: Path to the original tclp_market_share.csv
        output_file: Path to write the interpolated CSV
    """
    df = pd.read_csv(input_file)
    df["Year"] = df["Year"].astype(int)
    df = df.set_index("Year")

    # Determine the full year range from the data itself
    start_year = df.index.min()
    end_year = df.index.max()

    # Reindex to include every year, filling gaps with NaN
    df = df.reindex(range(start_year, end_year + 1))

    # Linear interpolation for all columns
    df = df.interpolate(method="linear")

    # Round to 2 decimal places
    df = df.round(2)

    # Reset index so Year becomes a column again
    df = df.reset_index()
    df = df.rename(columns={"index": "Year"})

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Written {len(df)} rows ({start_year}–{end_year}) to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Linearly interpolate TCLP market share data by year."
    )
    parser.add_argument(
        "--input",
        default="policy_regulation/tclp_market_share.csv",
        help="Path to the original tclp_market_share.csv",
    )
    parser.add_argument(
        "--output",
        default="policy_regulation/tclp_market_share_interpolated.csv",
        help="Path for the interpolated output CSV",
    )
    args = parser.parse_args()

    generate_tclp_market_share(
        input_file=args.input,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
