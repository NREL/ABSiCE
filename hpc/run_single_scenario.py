# -*- coding: utf-8 -*-
"""
TORC/Slurm entry point: run a single (landfill_set × ratio) scenario.

Each TORC job calls this script once.  It prepares the scaled cost files for
the requested scenario and then runs n_runs model iterations sequentially.

Usage (local smoke test):
    python hpc/run_single_scenario.py \
        --landfill-set all_landfills \
        --ratio 1.0 \
        --n-runs 2 \
        --n-steps 2

Usage (via TORC parameter expansion):
    python hpc/run_single_scenario.py \
        --landfill-set {landfill_set} \
        --ratio {ratio} \
        --n-runs 100
"""

import argparse
import sys
from pathlib import Path

# ── Make workspace root importable regardless of working directory ────────────
_HPC_DIR: Path = Path(__file__).parent.resolve()
_WORKSPACE_DIR: Path = _HPC_DIR.parent
if str(_WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_DIR))

# ── Import shared constants and helpers from run_rtn_scenarios ────────────────
# ProcessPoolExecutor is guarded inside main() so these imports are safe.
from run_rtn_scenarios import (  # noqa: E402
    _LANDFILL_SETS,
    _RESULTS_BASE,
    _HAZARDOUS_LANDFILL_FILE,
    _USPVDB_FILE,
    _build_suffix,
    _prepare_cost_files,
    _run_scenario,
)
from scale_recycling_cost import _BASELINE_RECYCLING_RATE_PER_KG  # noqa: E402
from utils import TIMESTEP  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for a single scenario run.

    Returns:
    argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run one (landfill_set × ratio) scenario for TORC/Slurm dispatch. "
            "Prepares scaled cost files then executes n_runs model iterations."
        )
    )
    parser.add_argument(
        "--landfill-set",
        required=True,
        choices=list(_LANDFILL_SETS.keys()),
        metavar="SET",
        help=(
            "Landfill set to run. "
            f"Choices: {list(_LANDFILL_SETS.keys())}"
        ),
    )
    cost_group = parser.add_mutually_exclusive_group(required=True)
    cost_group.add_argument(
        "--ratio",
        type=float,
        metavar="FLOAT",
        help=(
            "Cost scale multiplier (e.g. 0.75 for -25%%, 1.0 for baseline, "
            "1.25 for +25%%). Values < 1.0 are allowed."
        ),
    )
    cost_group.add_argument(
        "--cost-rate",
        type=float,
        metavar="RATE",
        help=(
            f"Absolute recycling cost rate in $/kg "
            f"(e.g. 0.40 for baseline, 0.30 for -25%%, 0.50 for +25%%). "
            f"Baseline is {_BASELINE_RECYCLING_RATE_PER_KG} $/kg. "
            "Only valid with --cost-component recycling."
        ),
    )
    parser.add_argument(
        "--cost-component",
        default="recycling",
        choices=["recycling", "transport"],
        metavar="COMPONENT",
        help="Cost column to scale: 'recycling' (default) or 'transport'.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=100,
        metavar="N",
        help="Number of model runs (default: 100).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=11,
        metavar="N",
        help="Simulation length in years (default: 11).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Prepare cost files and run all model iterations for a single scenario.

    Returns:
    None
    """
    args: argparse.Namespace = _parse_args()

    # Resolve ratio — convert absolute cost rate if provided
    if args.cost_rate is not None:
        if args.cost_component == "transport":
            print(
                "Error: --cost-rate is only valid with --cost-component recycling. "
                "Transport cost varies per route and cannot be specified as a single rate.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)
        ratio: float = args.cost_rate / _BASELINE_RECYCLING_RATE_PER_KG
    else:
        ratio = args.ratio

    set_config: dict = _LANDFILL_SETS[args.landfill_set]
    suffix: str = _build_suffix(ratio, args.cost_component)
    output_dir: Path = _RESULTS_BASE / f"{set_config['results_prefix']}{suffix}"

    rate_info: str = (
        f"cost_rate={args.cost_rate} $/kg (ratio={ratio:g})"
        if args.cost_rate is not None
        else f"ratio={ratio:g}"
    )
    print(
        f"\n=== Scenario: {args.landfill_set}{suffix} ===\n"
        f"  {rate_info}  cost_component={args.cost_component}\n"
        f"  n_runs={args.n_runs}  n_steps={args.n_steps}\n"
        f"  output_dir={output_dir}\n",
        flush=True,
    )

    # ── Phase 1: prepare scaled cost CSVs ────────────────────────────────────
    print("-- Preparing cost files ...", flush=True)
    recycling_filename: str
    landfill_filename: str
    recycling_filename, landfill_filename = _prepare_cost_files(
        set_config=set_config,
        ratio=ratio,
        suffix=suffix,
        uspvdb_file=str(_USPVDB_FILE),
        cost_component=args.cost_component,
    )
    print(
        f"  recycling file : {recycling_filename}\n"
        f"  landfill file  : {landfill_filename}\n",
        flush=True,
    )

    # ── Phase 2: run model iterations ────────────────────────────────────────
    print("-- Running model iterations ...", flush=True)
    _run_scenario(
        recycling_filename=recycling_filename,
        landfill_filename=landfill_filename,
        output_dir=str(output_dir),
        n_runs=args.n_runs,
        n_steps_years=args.n_steps,
        timestep_value=TIMESTEP.QUARTERLY.value,
    )

    print(f"\n=== Done: {args.landfill_set}{suffix} ===\n", flush=True)


if __name__ == "__main__":
    main()
