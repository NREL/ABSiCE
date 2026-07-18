"""
Run aggregate_consumer_results for every scenario subfolder under a base directory.

A "scenario folder" is any directory (at any depth) that contains at least one
file matching Results_agents_consumers_run_*.csv.

Usage:
    python run_aggregate_all_scenarios.py [--base_dir DIR] [--columns COL1 COL2 ...]

Defaults:
    base_dir : results/RTN_Iteration_3
    columns  : Waste Repair (Kg) Waste Sell (Kg) Waste Recycle (Kg)
               Waste Landfill (Kg) Waste Hoard (Kg)
"""

import argparse
import os
import re

from plotting import aggregate_and_save_consumer_results, plot_consumer_waste_by_site

DEFAULT_COLUMNS = [
    "Waste Repair (Kg)",
    "Waste Sell (Kg)",
    "Waste Recycle (Kg)",
    "Waste Landfill (Kg)",
    "Waste Hoard (Kg)",
]

_CONSUMER_FILE_PATTERN = re.compile(r"Results_agents_consumers_run_\d+\.csv")


def _is_scenario_dir(path: str) -> bool:
    """Return True if *path* contains at least one consumer results CSV."""
    try:
        return any(_CONSUMER_FILE_PATTERN.match(f) for f in os.listdir(path))
    except PermissionError:
        return False


def find_scenario_dirs(base_dir: str) -> list[str]:
    """Recursively collect all scenario directories under *base_dir*."""
    scenario_dirs: list[str] = []
    for dirpath, dirnames, _ in os.walk(base_dir):
        dirnames.sort()  # deterministic order
        if _is_scenario_dir(dirpath):
            scenario_dirs.append(dirpath)
            dirnames.clear()  # don't descend further; results live at this level
    return scenario_dirs


def run_aggregate(base_dir: str, columns: list[str]) -> None:
    scenario_dirs = find_scenario_dirs(base_dir)

    if not scenario_dirs:
        print(f"No scenario directories found under {base_dir}")
        return

    print(f"Found {len(scenario_dirs)} scenario director(ies):\n")
    for d in scenario_dirs:
        print(f"  {d}")
    print()

    for i, scenario_dir in enumerate(scenario_dirs, start=1):
        n_files = sum(1 for f in os.listdir(scenario_dir) if _CONSUMER_FILE_PATTERN.match(f))
        print(f"[{i}/{len(scenario_dirs)}] Processing: {scenario_dir}  ({n_files} iteration file(s))")
        try:
            aggregate_and_save_consumer_results(results_dir=scenario_dir, columns=columns)
            plot_consumer_waste_by_site(results_dir=scenario_dir, value_columns=columns)
            print(f"  Done.\n")
        except Exception as exc:
            print(f"  ERROR: {exc}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate consumer results for all scenario subfolders."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="results/RTN_Iteration_3",
        help="Base directory to search for scenario subfolders (default: results/RTN_Iteration_3).",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="*",
        default=DEFAULT_COLUMNS,
        help="Waste columns to aggregate and plot (default: all five waste types).",
    )

    args = parser.parse_args()
    run_aggregate(base_dir=args.base_dir, columns=args.columns)
