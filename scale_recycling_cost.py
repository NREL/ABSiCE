# -*- coding: utf-8 -*-
"""
Scale a cost column in RTN shipment CSV files by a given multiplier,
recalculate the total cost column, and save the result with a suffix
encoding the scale factor.

Usage:
    python scale_recycling_cost.py \
        --files /path/to/file1.csv /path/to/file2.csv \
        --scale 1.05 \
        [--cost-col "RecyclingCost_$"] \
        [--transport-col "TransportCost_$"] \
        [--total-col "TotalCost_$"]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Baseline recycling rate in $/kg, consistent across all shipment rows.
# RecyclingCost_$ = Shipped_kg × _BASELINE_RECYCLING_RATE_PER_KG.
# Used to convert an absolute cost rate to a scale multiplier.
_BASELINE_RECYCLING_RATE_PER_KG: float = 0.40


def _build_suffix(scale: float) -> str:
    """
    Build a filename suffix from the scale factor.

    Parameters:
    scale (float): Multiplier to apply (e.g. 1.05 for +5%, 0.9 for -10%).

    Returns:
    str: Suffix string, e.g. '_1.05' or '_neg0.9'.
    """
    if scale >= 1.0:
        return f"_{scale:g}"
    else:
        return f"_neg{scale:g}"


def _scale_file(
    file_path: Path,
    scale: float,
    cost_col: str,
    transport_col: str,
    total_col: str,
    file_suffix: str | None = None,
) -> Path:
    """
    Scale the cost column of a single CSV file and save the result.

    Parameters:
    file_path (Path): Path to the input CSV file.
    scale (float): Multiplier to apply to the cost column.
    cost_col (str): Name of the column to scale.
    transport_col (str): Name of the other cost column used to recalculate total.
    total_col (str): Name of the total cost column to recalculate.
    file_suffix (str | None): Override the output filename suffix. If None,
        the suffix is derived from scale via _build_suffix (default behaviour).

    Returns:
    Path: Path to the saved output file.
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

    suffix: str = file_suffix if file_suffix is not None else _build_suffix(scale)
    out_path: Path = file_path.with_stem(file_path.stem + suffix)
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    """
    Parse CLI arguments and process each input file.

    Returns:
    None
    """
    parser = argparse.ArgumentParser(
        description=(
            "Scale a cost column in RTN shipment CSV files by a multiplier, "
            "recalculate the total cost, and save with a suffix."
        )
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        metavar="FILE",
        help="One or more input CSV file paths.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        required=True,
        help=(
            "Scale factor as a multiplier "
            "(e.g. 1.05 for +5%%, 0.9 for -10%%)."
        ),
    )
    parser.add_argument(
        "--cost-col",
        default="RecyclingCost_$",
        metavar="COL",
        help="Name of the cost column to scale (default: 'RecyclingCost_$').",
    )
    parser.add_argument(
        "--transport-col",
        default="TransportCost_$",
        metavar="COL",
        help=(
            "Name of the transport cost column used in total recalculation "
            "(default: 'TransportCost_$')."
        ),
    )
    parser.add_argument(
        "--total-col",
        default="TotalCost_$",
        metavar="COL",
        help=(
            "Name of the total cost column to recalculate "
            "(default: 'TotalCost_$')."
        ),
    )

    args = parser.parse_args()

    if args.scale <= 0:
        parser.error("--scale must be a positive number.")

    errors: list[str] = []
    for file_str in args.files:
        file_path = Path(file_str)
        if not file_path.is_file():
            errors.append(f"File not found: {file_path}")
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    for file_str in args.files:
        file_path = Path(file_str)
        try:
            out_path = _scale_file(
                file_path=file_path,
                scale=args.scale,
                cost_col=args.cost_col,
                transport_col=args.transport_col,
                total_col=args.total_col,
            )
            print(f"Saved: {out_path}")
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
