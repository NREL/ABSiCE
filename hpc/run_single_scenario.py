# -*- coding: utf-8 -*-
"""
TORC/Slurm entry point: run a single scenario via ``hpc.scenario_runner``.

Each TORC job calls this script once. It resolves scenario overrides from
CLI flags and dispatches them to :func:`hpc.scenario_runner.run_scenario`,
which builds the resolved config, prepares any RTN cost files, and runs
``n_runs`` replicate model iterations through ``absice.runner.run_batch``.

The same script drives both RTN and non-RTN runs (``--rtn`` / ``--no-rtn``);
see ``hpc/scenario_runner.py`` for the two modes' semantics.

Usage (local smoke test, non-RTN):
    python hpc/run_single_scenario.py \\
        --landfill-set all_landfills \\
        --ratio 1.0 \\
        --no-rtn \\
        --n-runs 1 \\
        --n-steps 1 \\
        --results-base /tmp/smoke

Usage (RTN mode, absolute $/kg cost rate):
    python hpc/run_single_scenario.py \\
        --landfill-set all_landfills \\
        --cost-rate 12.02 \\
        --rtn \\
        --recycle-rate 0.20 \\
        --att-mean 0.60 \\
        --n-runs 1 \\
        --n-steps 1 \\
        --results-base /tmp/smoke

Usage (via TORC parameter expansion):
    python hpc/run_single_scenario.py \\
        --landfill-set {landfill_set} \\
        --cost-rate {cost_rate} \\
        --rtn \\
        --recycle-rate {recycle_rate} \\
        --n-runs 100

Transport-cost sensitivity is out of scope this session:
``--cost-component transport`` parses but is rejected downstream by
``hpc.scenario_runner.run_scenario``.
"""

import argparse
import sys
from pathlib import Path

# ── Make workspace root importable regardless of working directory ────────────
_HPC_DIR: Path = Path(__file__).parent.resolve()
_WORKSPACE_DIR: Path = _HPC_DIR.parent
if str(_WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_DIR))

from hpc import cost_scaling  # noqa: E402
from hpc.scenario_runner import run_scenario  # noqa: E402

# Default attitude mean — matches hpc/rtn_base.yaml's
# tpb.att_distrib_param_eol[0] baseline. Runs that do not pass --att-mean
# use this value and the output path is unchanged (no att_mean_X.XX
# subdirectory inserted; see hpc/scenario_runner.py's own base-config
# comparison, which is config-driven and does not depend on this constant).
_ATT_MEAN_DEFAULT: float = 0.515

# Default results base, mirroring the old kwargs-runner's results/ layout.
_RESULTS_BASE_DEFAULT: Path = _WORKSPACE_DIR / "results" / "hpc_scenarios"


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for a single scenario run.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run one scenario for TORC/Slurm dispatch via "
            "hpc.scenario_runner.run_scenario. Runs whether or not RTN "
            "is enabled (--rtn / --no-rtn)."
        )
    )
    parser.add_argument(
        "--landfill-set",
        required=True,
        choices=list(cost_scaling.LANDFILL_SETS.keys()),
        metavar="SET",
        help=(
            "Landfill set to run. "
            f"Choices: {list(cost_scaling.LANDFILL_SETS.keys())}"
        ),
    )
    cost_group = parser.add_mutually_exclusive_group(required=True)
    cost_group.add_argument(
        "--ratio",
        type=float,
        metavar="FLOAT",
        help=(
            "Recycling-cost scale multiplier (e.g. 0.75 for -25%%, 1.0 for "
            "baseline, 1.25 for +25%%). Works in both RTN and non-RTN mode."
        ),
    )
    cost_group.add_argument(
        "--cost-rate",
        type=float,
        metavar="RATE",
        help=(
            "Absolute RTN recycling cost rate in $/kg (e.g. 0.40 for "
            f"baseline, {cost_scaling.BASELINE_RECYCLING_RATE_PER_KG} "
            "is the anchor). RTN-specific: requires --rtn. Converted to a "
            "ratio via hpc.cost_scaling.ratio_from_cost_rate."
        ),
    )
    parser.add_argument(
        "--cost-component",
        default="recycling",
        choices=["recycling", "transport"],
        metavar="COMPONENT",
        help=(
            "Cost column to scale. Only 'recycling' (default) is "
            "supported this session; 'transport' parses but is rejected "
            "by run_scenario (out of scope)."
        ),
    )
    rtn_group = parser.add_mutually_exclusive_group()
    rtn_group.add_argument(
        "--rtn",
        dest="rtn",
        action="store_true",
        default=None,
        help=(
            "Run in RTN mode (config.data_source.rtn=True): recycling-cost "
            "(and, unscaled, landfill-cost) CSVs are read from RTN/, "
            "requiring RTN_Data for any --ratio/--cost-rate != 1.0."
        ),
    )
    rtn_group.add_argument(
        "--no-rtn",
        dest="rtn",
        action="store_false",
        default=None,
        help=(
            "Run in non-RTN mode (config.data_source.rtn=False): the "
            "triangular recycling-cost lever (config.cost."
            "original_recycling_cost) is scaled directly. No RTN_Data "
            "filesystem access."
        ),
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
    parser.add_argument(
        "--results-base",
        type=Path,
        default=_RESULTS_BASE_DEFAULT,
        metavar="DIR",
        help=(
            "Base directory for scenario output folders "
            f"(default: {_RESULTS_BASE_DEFAULT}). "
            "Override on the cluster to point to a cluster-local path."
        ),
    )
    parser.add_argument(
        "--recycle-rate",
        type=float,
        default=0.10,
        metavar="RATE",
        help=(
            "Initial recycling EoL rate (0.0-1.0). The delta vs the base "
            "config's own baseline recycle rate is subtracted from the "
            "landfill rate so all rates sum to 1. Default: 0.10 "
            "(hpc/rtn_base.yaml baseline). Results go into a "
            "recycle_rate_<XX>pct/ subdirectory."
        ),
    )
    parser.add_argument(
        "--att-mean",
        type=float,
        default=_ATT_MEAN_DEFAULT,
        metavar="FLOAT",
        help=(
            "Mean of the bounded-normal attitude-toward-EoL-recycling "
            f"distribution (att_distrib_param_eol[0]). Default: "
            f"{_ATT_MEAN_DEFAULT}. Values away from the base config's own "
            "default insert an att_mean_X.XX/ subdirectory into the "
            "output path so calibration sweeps stay isolated."
        ),
    )
    parser.add_argument(
        "--att-std",
        type=float,
        default=0.1,
        metavar="FLOAT",
        help=(
            "Standard deviation of the attitude distribution "
            "(att_distrib_param_eol[1]). Default: 0.1."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of parallel worker processes for run_batch. Defaults "
            "to the available CPU count."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Resolve scenario overrides and dispatch to ``run_scenario``.

    Returns
    -------
    None
    """
    args: argparse.Namespace = _parse_args()

    # --rtn / --no-rtn default: whatever hpc/rtn_base.yaml has
    # (data_source.rtn) when neither flag is passed.
    rtn: bool
    if args.rtn is None:
        from absice.schemas.simulation_config import SimulationConfig
        from hpc.scenario_runner import _BASE_CONFIG_PATH

        rtn = SimulationConfig.from_yaml(_BASE_CONFIG_PATH).data_source.rtn
    else:
        rtn = args.rtn

    # Resolve ratio — convert absolute cost rate if provided.
    # --cost-rate is $/kg and only meaningful for RTN mode (Gate A R4):
    # reject it outright when rtn is disabled, convert it when rtn is
    # enabled.
    if args.cost_rate is not None:
        if args.cost_component == "transport":
            print(
                "Error: --cost-rate is only valid with --cost-component "
                "recycling. Transport cost varies per route and cannot be "
                "specified as a single rate.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)
        if not rtn:
            print(
                "Error: --cost-rate is RTN-specific ($/kg shipment cost) "
                "and is not meaningful in non-RTN mode, where recycling "
                "cost is a triangular $/functional-unit distribution. "
                "Pass --ratio instead, or add --rtn to run in RTN mode.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)
        ratio: float = cost_scaling.ratio_from_cost_rate(args.cost_rate)
    else:
        ratio = args.ratio

    rate_info: str = (
        f"cost_rate={args.cost_rate} $/kg (ratio={ratio:g})"
        if args.cost_rate is not None
        else f"ratio={ratio:g}"
    )
    print(
        f"\n=== Scenario: {args.landfill_set} ===\n"
        f"  rtn={rtn}  {rate_info}  cost_component={args.cost_component}\n"
        f"  recycle_rate={args.recycle_rate:.0%}\n"
        f"  att_distrib_param_eol=[{args.att_mean}, {args.att_std}]\n"
        f"  n_runs={args.n_runs}  n_steps={args.n_steps}\n"
        f"  results_base={args.results_base}\n",
        flush=True,
    )

    output_paths = run_scenario(
        landfill_set=args.landfill_set,
        ratio=ratio,
        cost_component=args.cost_component,
        n_runs=args.n_runs,
        n_steps=args.n_steps,
        results_base=args.results_base,
        recycle_rate=args.recycle_rate,
        att_mean=args.att_mean,
        att_std=args.att_std,
        rtn=rtn,
        workers=args.workers,
    )

    print(
        f"\n=== Done: {args.landfill_set} ({len(output_paths)} output file(s)) ===\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
