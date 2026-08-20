# -*- coding: utf-8 -*-
"""
Analyse att_distrib_param_eol calibration sweep results.

Reads Results_agents_consumers_run_*.csv files from
results/att_calibration/att_mean_X.XX/**/ and computes the mean recycling
rate for each att_mean value.  Recycling rate is defined as:

    recycling_rate = sum("Waste Recycle (Kg)") / sum(all 5 waste columns)

aggregated across all agents, all timesteps, and all replicates.

Output layout produced by hpc/run_single_scenario.py (via
hpc/scenario_runner.py::run_scenario) is:

    results_base/
      att_mean_X.XX/                          # only when att_mean != base
        recycle_rate_XXpct/
          <results_prefix><suffix>/
            Results_agents_consumers_run_<id>.csv
            Results_model_run_<id>.csv
            scenario_config.yaml

This differs from the old kwargs-runner scheme only in the innermost
directory name (a landfill_set-derived prefix + ratio/cost-rate suffix
instead of a bare "RTN_run_..." name); the recursive glob below matches
either shape unchanged. RTN mode is not required — this analyzer has no
RTN-specific assumptions of its own.

Outputs:
  results/att_calibration/calibration_curve.csv
  results/att_calibration/calibration_curve.png

Usage:
    python hpc/analyze_att_calibration.py
    python hpc/analyze_att_calibration.py --results-base results/att_calibration
    python hpc/analyze_att_calibration.py --target-rate 0.20 --plot
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

# ── Resolve workspace root so the script can be run from any working dir ──────
_HPC_DIR: Path = Path(__file__).parent.resolve()
_WORKSPACE_DIR: Path = _HPC_DIR.parent

_WASTE_COLUMNS: list[str] = [
    "Waste Repair (Kg)",
    "Waste Sell (Kg)",
    "Waste Recycle (Kg)",
    "Waste Landfill (Kg)",
    "Waste Hoard (Kg)",
]


def _recycling_rate_from_consumers_csv(path: str) -> float:
    """
    Compute recycling rate from one Results_agents_consumers_run_*.csv file.

    Returns Waste Recycle / total waste summed across all agents and
    timesteps. Returns NaN if the file is empty or missing required
    columns.
    """
    df: pd.DataFrame = pd.read_csv(path)
    present: list[str] = [c for c in _WASTE_COLUMNS if c in df.columns]
    if "Waste Recycle (Kg)" not in present or not present:
        return float("nan")
    total_recycle: float = df["Waste Recycle (Kg)"].sum()
    total_waste: float = df[present].sum().sum()
    if total_waste == 0:
        return float("nan")
    return total_recycle / total_waste


def _collect_results(results_base: Path) -> pd.DataFrame:
    """
    Walk results_base/att_mean_X.XX/ trees and compute per-run recycling
    rates.

    Returns a DataFrame with columns: att_mean, run_idx, recycling_rate.
    """
    att_mean_dir_re = "att_mean_*"
    records: list[dict] = []

    for att_dir in sorted(results_base.glob(att_mean_dir_re)):
        if not att_dir.is_dir():
            continue
        try:
            att_mean: float = float(att_dir.name.split("_")[-1])
        except ValueError:
            continue

        # Results are nested inside recycle_rate_XXpct/<results_prefix>
        # <suffix>/ subdirs (hpc/scenario_runner.py's output-label scheme);
        # the recursive glob matches regardless of the innermost directory
        # name.
        consumer_csvs: list[str] = sorted(
            glob.glob(
                str(att_dir / "**" / "Results_agents_consumers_run_*.csv"),
                recursive=True,
            )
        )
        if not consumer_csvs:
            print(f"  [warn] No consumer CSVs found under {att_dir}", flush=True)
            continue

        for csv_path in consumer_csvs:
            run_idx: int
            try:
                run_idx = int(
                    Path(csv_path).stem.split("Results_agents_consumers_run_")[-1]
                )
            except ValueError:
                run_idx = -1

            rate: float = _recycling_rate_from_consumers_csv(csv_path)
            records.append(
                {"att_mean": att_mean, "run_idx": run_idx, "recycling_rate": rate}
            )

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise att_distrib_param_eol calibration sweep results."
    )
    parser.add_argument(
        "--results-base",
        type=Path,
        default=_WORKSPACE_DIR / "results" / "att_calibration",
        metavar="DIR",
        help=(
            "Base directory written by the calibration TORC run "
            "(default: results/att_calibration)."
        ),
    )
    parser.add_argument(
        "--target-rate",
        type=float,
        default=0.20,
        metavar="RATE",
        help="Target recycling rate for calibration (default: 0.20).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Save a calibration curve PNG (default: True).",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip the PNG output.",
    )
    args = parser.parse_args()

    results_base: Path = Path(args.results_base).resolve()
    if not results_base.exists():
        print(f"Error: results directory not found: {results_base}", file=sys.stderr)
        sys.exit(1)

    print(f"\nCollecting results from {results_base} ...", flush=True)
    df: pd.DataFrame = _collect_results(results_base)

    if df.empty:
        print("No results found. Check that the TORC run completed successfully.")
        sys.exit(1)

    # ── Aggregate: mean +/- std recycling rate per att_mean ────────────────
    summary: pd.DataFrame = (
        df.groupby("att_mean")["recycling_rate"]
        .agg(
            n_runs="count",
            mean_recycling_rate="mean",
            std_recycling_rate="std",
            min_recycling_rate="min",
            max_recycling_rate="max",
        )
        .reset_index()
        .sort_values("att_mean")
    )

    # ── Save CSV ────────────────────────────────────────────────────────────
    csv_out: Path = results_base / "calibration_curve.csv"
    summary.to_csv(csv_out, index=False)
    print(f"\nCalibration curve saved to {csv_out}")

    # ── Print summary table ──────────────────────────────────────────────────
    print("\natt_mean  | n_runs | mean_rate | std_rate | target_delta")
    print("-" * 60)
    for _, row in summary.iterrows():
        delta: float = row["mean_recycling_rate"] - args.target_rate
        print(
            f"  {row['att_mean']:.2f}    |  {row['n_runs']:4.0f}  "
            f"| {row['mean_recycling_rate']:9.4f} | {row['std_recycling_rate']:8.4f} "
            f"| {delta:+.4f}"
        )

    # ── Identify best calibration value ────────────────────────────────────
    best_idx: int = (summary["mean_recycling_rate"] - args.target_rate).abs().idxmin()
    best_row = summary.loc[best_idx]
    print(
        f"\nBest att_mean = {best_row['att_mean']:.2f}  "
        f"(mean recycling rate = {best_row['mean_recycling_rate']:.4f}, "
        f"target = {args.target_rate:.2f})"
    )
    print(
        "\nTo apply: set tpb.att_distrib_param_eol in the relevant config "
        f"YAML to:\n    [{best_row['att_mean']:.2f}, 0.1]"
    )

    if not args.plot:
        return

    # ── Plot calibration curve ──────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("\n[warn] matplotlib not available -- skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        summary["att_mean"],
        summary["mean_recycling_rate"],
        yerr=summary["std_recycling_rate"],
        marker="o",
        capsize=4,
        label="Mean recycling rate +/- 1 std",
    )
    ax.axhline(
        args.target_rate,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Target = {args.target_rate:.0%}",
    )
    ax.axvline(
        best_row["att_mean"],
        color="green",
        linestyle=":",
        linewidth=1.2,
        label=f"Best att_mean = {best_row['att_mean']:.2f}",
    )

    ax.set_xlabel("att_distrib_param_eol mean")
    ax.set_ylabel("Simulated recycling rate")
    ax.set_title(
        "Calibration: recycling rate vs attitude distribution mean"
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    ax.grid(True, alpha=0.3)

    png_out: Path = results_base / "calibration_curve.png"
    fig.savefig(png_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {png_out}")


if __name__ == "__main__":
    main()
