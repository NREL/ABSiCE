# -*- coding: utf-8 -*-
"""
Run RTN scenario simulations in parallel across recycling cost ratios.

Pipeline per (landfill_set, ratio) combination:
  1. Scale the source shipment CSV by ratio        → RTN_Data/Round_3/*{suffix}.csv
  2. Run generate_recycling_costs on scaled file   → ABSiCE/RTN/*{suffix}.csv
  3. Run N model runs with all calibrated params   → results/RTN_Iteration_3/*{suffix}/

Both landfill sets (all_landfills, true_landfills) are processed for every ratio.
Scenarios are dispatched to a ProcessPoolExecutor (one process per scenario).

Usage:
    python run_rtn_scenarios.py --ratios 0.9 1.0 1.05
    python run_rtn_scenarios.py --ratios=0.75 1.0 1.25 --n-runs 10 --n-steps 11
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ── Workspace-relative constants ─────────────────────────────────────────────
_WORKSPACE_DIR: Path = Path(__file__).parent.resolve()
_RTN_DIR: Path = _WORKSPACE_DIR / "RTN"
_RTN_DATA_DIR: Path = next(
    p for p in [
        Path("/projects/pvabm/pghosh/RTN_Data/Round_3"),  # HPC (Kestrel)
        Path("/Users/pghosh/SOLAR/RTN_Data/Round_3"),     # macOS local
    ]
    if p.exists()
)
_USPVDB_FILE: Path = (
    _WORKSPACE_DIR / "USPVDB" / "uspvdb_v3_0_20250430_with_pca.xlsx"
)
_RESULTS_BASE: Path = _WORKSPACE_DIR / "results" / "RTN_Iteration_3"

# Calibrated run parameters — mirrors run_model(10, 11, QUARTERLY) in
# ABM_CE_PV_MultipleRun.py
_N_RUNS_DEFAULT: int = 10
_N_STEPS_DEFAULT: int = 11  # simulation years; multiplied by timestep below

# Fixed model parameters from j < 10 block in ABM_CE_PV_MultipleRun.py
_FIXED_MODEL_PARAMS: dict = {
    "hazardous_waste_regulation_enabled": False,
    "landfill_solar_waste_acceptance_ratio": 1.0,
    "calculate_distances": False,
    "model_states": ["TX", "AZ", "NV", "NM"],
    "solar_cycle": False,
    "rtn": True,
}

# Per landfill-set file mapping.  Recycling data varies with recycling cost ratio;
# landfill data varies with transport cost ratio; hazardous landfill data is fixed.
_LANDFILL_SETS: dict = {
    "all_landfills": {
        "shipment_file": _RTN_DATA_DIR / "shipments_recycle_alllandfills.csv",
        "landfill_shipment_file": _RTN_DATA_DIR / "shipments_landfill_alllandfills.csv",
        "recycling_file_prefix": "RecyclingCostsbyYearAllLandfills_3",
        "landfill_file": "LandfillCostsbyYearAllLandfills_3.csv",
        "results_prefix": "RTN_run_all_landfills",
    },
    "true_landfills": {
        "shipment_file": _RTN_DATA_DIR / "shipments_recycle_truelandfills.csv",
        "landfill_shipment_file": _RTN_DATA_DIR / "shipments_landfill_truelandfills.csv",
        "recycling_file_prefix": "RecyclingCostsbyYearTrueLandfills_3",
        "landfill_file": "LandfillCostsbyYearTrueLandfills_3.csv",
        "results_prefix": "RTN_run_true_landfills",
    },
}
_HAZARDOUS_LANDFILL_FILE: str = "Landfills_data_SA.csv"


# ── File preparation helpers ──────────────────────────────────────────────────

def _build_suffix(ratio: float, cost_component: str = "recycling") -> str:
    """
    Build a filename suffix from the scale ratio and cost component.

    Parameters:
    ratio (float): Multiplier (e.g. 1.05 for +5%, 0.9 for -10%).
    cost_component (str): 'recycling' (default) or 'transport'. Transport
        scenarios get a '_transport' infix to avoid collisions with recycling
        scenario folders at the same ratio.

    Returns:
    str: Suffix string, e.g. '_1.05', '_neg0.9', '_transport_1.05',
        '_transport_neg0.9'.
    """
    infix: str = "_transport" if cost_component == "transport" else ""
    if ratio >= 1.0:
        return f"{infix}_{ratio:g}"
    return f"{infix}_neg{ratio:g}"


def _prepare_cost_files(
    set_config: dict,
    ratio: float,
    suffix: str,
    uspvdb_file: str,
    cost_component: str = "recycling",
) -> tuple[str, str]:
    """
    Scale shipment files and generate processed cost CSVs for the model.

    For ratio == 1.0, no files are generated; the original base files are used.

    For cost_component == 'recycling': scales RecyclingCost_$ in the recycling
    shipment file only; the landfill cost file is unchanged.

    For cost_component == 'transport': scales TransportCost_$ in both the
    recycling and landfill shipment files, then regenerates both cost CSVs.
    Transport-scaled intermediates use suffix as their filename suffix so they
    cannot collide with recycling-scaled files at the same ratio.

    Parameters:
    set_config (dict): Landfill-set configuration entry from _LANDFILL_SETS.
    ratio (float): Scale multiplier to apply to the chosen cost column.
    suffix (str): Filename suffix derived from ratio and cost_component.
    uspvdb_file (str): Path to the USPVDB Excel file.
    cost_component (str): 'recycling' or 'transport'.

    Returns:
    tuple[str, str]: (recycling_filename, landfill_filename) — filenames
        (not full paths) of the cost CSVs in ABSiCE/RTN/.
    """
    base_recycling_filename: str = f"{set_config['recycling_file_prefix']}.csv"
    base_landfill_filename: str = set_config["landfill_file"]

    if abs(ratio - 1.0) < 1e-9:
        return base_recycling_filename, base_landfill_filename

    from scale_recycling_cost import _scale_file
    from generate_recycling_costs import generate_recycling_costs
    from generate_landfill_costs import generate_landfill_costs

    if cost_component == "recycling":
        # Scale RecyclingCost_$ in the recycling shipment file only.
        scaled_shipment: Path = _scale_file(
            file_path=set_config["shipment_file"],
            scale=ratio,
            cost_col="RecyclingCost_$",
            transport_col="TransportCost_$",
            total_col="TotalCost_$",
        )
        print(f"  Scaled recycling shipment: {scaled_shipment.name}")

        recycling_filename: str = (
            f"{set_config['recycling_file_prefix']}{suffix}.csv"
        )
        recycling_output: str = str(_RTN_DIR / recycling_filename)
        if Path(recycling_output).exists():
            print(f"  Recycling file already exists, skipping: {recycling_filename}")
        else:
            generate_recycling_costs(
                shipments_file=str(scaled_shipment),
                uspvdb_file=uspvdb_file,
                output_file=recycling_output,
            )
        return recycling_filename, base_landfill_filename

    else:  # transport
        # Scale TransportCost_$ in both shipment files.  Use suffix as the
        # file_suffix so transport-scaled intermediates (e.g.
        # shipments_recycle_alllandfills_transport_1.05.csv) are distinct from
        # recycling-scaled intermediates at the same ratio.
        scaled_recycle_shipment: Path = _scale_file(
            file_path=set_config["shipment_file"],
            scale=ratio,
            cost_col="TransportCost_$",
            transport_col="RecyclingCost_$",
            total_col="TotalCost_$",
            file_suffix=suffix,
        )
        print(f"  Scaled recycling shipment: {scaled_recycle_shipment.name}")

        scaled_landfill_shipment: Path = _scale_file(
            file_path=set_config["landfill_shipment_file"],
            scale=ratio,
            cost_col="TransportCost_$",
            transport_col="DisposalCost_$",
            total_col="TotalCost_$",
            file_suffix=suffix,
        )
        print(f"  Scaled landfill shipment: {scaled_landfill_shipment.name}")

        # Generate recycling cost file
        recycling_filename: str = (
            f"{set_config['recycling_file_prefix']}{suffix}.csv"
        )
        recycling_output: str = str(_RTN_DIR / recycling_filename)
        if Path(recycling_output).exists():
            print(f"  Recycling file already exists, skipping: {recycling_filename}")
        else:
            generate_recycling_costs(
                shipments_file=str(scaled_recycle_shipment),
                uspvdb_file=uspvdb_file,
                output_file=recycling_output,
            )

        # Generate landfill cost file with transport-specific suffix
        landfill_prefix: str = set_config["landfill_file"].replace(".csv", "")
        landfill_filename: str = f"{landfill_prefix}{suffix}.csv"
        landfill_output: str = str(_RTN_DIR / landfill_filename)
        if Path(landfill_output).exists():
            print(f"  Landfill file already exists, skipping: {landfill_filename}")
        else:
            generate_landfill_costs(
                shipments_file=str(scaled_landfill_shipment),
                uspvdb_file=uspvdb_file,
                output_file=landfill_output,
            )

        return recycling_filename, landfill_filename


# ── Worker function (runs in subprocess) ─────────────────────────────────────

def _run_scenario(
    recycling_filename: str,
    landfill_filename: str,
    output_dir: str,
    n_runs: int,
    n_steps_years: int,
    timestep_value: int,
) -> None:
    """
    Run n_runs model simulations for one (landfill_set, ratio) scenario.

    Saves Results_model_run_{j}.csv and Results_agents_consumers_run_{j}.csv
    per run.  All calibrated parameters from the j < 10 block of
    ABM_CE_PV_MultipleRun.py are applied.

    Parameters:
    recycling_filename (str): Filename of the recycling cost CSV (in RTN/).
    landfill_filename (str): Filename of the landfill cost CSV (in RTN/).
    output_dir (str): Absolute path to the results subfolder.
    n_runs (int): Number of model runs to perform.
    n_steps_years (int): Simulation length in years.
    timestep_value (int): TIMESTEP enum value (e.g. 4 for QUARTERLY).

    Returns:
    None
    """
    # Defer imports to subprocess — avoids triggering module-level code in the
    # main process and keeps each worker self-contained.
    from ABM_CE_PV_Model import ABM_CE_PV
    from ABM_CE_PV_ConsumerAgents import Consumers
    from utils import TIMESTEP

    timestep: TIMESTEP = TIMESTEP(timestep_value)
    number_steps: int = n_steps_years * timestep_value

    # Resolve output directory as absolute before any os.chdir can happen
    output_path: Path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scenario_label: str = output_path.name

    # PV_ICE._setPath calls os.chdir(self.path) which moves cwd to
    # PV_ICE/TEMP/PCA. ABM_CE_PV_Model uses Path().resolve() (cwd-relative)
    # to build baseline paths, so cwd must be reset to the workspace root
    # before every model instantiation.
    workspace_dir: str = str(Path(__file__).parent.resolve())

    for j in range(n_runs):
        os.chdir(workspace_dir)
        t0: float = time.time()

        model: ABM_CE_PV = ABM_CE_PV(
            seed=j,
            last_step=number_steps,
            hazardous_waste_regulation_enabled=False,
            landfill_solar_waste_acceptance_ratio=1.0,
            calculate_distances=False,
            model_states=["TX", "AZ", "NV", "NM"],
            solar_cycle=False,
            rtn=True,
            file_name={
                "Landfill data": landfill_filename,
                "Recycling data": recycling_filename,
                "Hazardous landfill data": _HAZARDOUS_LANDFILL_FILE,
            },
            timestep=timestep,
        )

        for _ in range(number_steps):
            model.step()

        results_model = model.datacollector.get_model_vars_dataframe()
        results_agents_consumers = (
            model.datacollector.get_agenttype_vars_dataframe(
                agent_type=Consumers
            )
        )
        results_agents_consumers.reset_index(inplace=True)
        results_agents_consumers.drop(columns=["Step", "AgentID"], inplace=True)

        results_model.to_csv(output_path / f"Results_model_run_{j}.csv")
        results_agents_consumers.to_csv(
            output_path / f"Results_agents_consumers_run_{j}.csv",
            index=False,
        )

        t1: float = time.time()
        print(
            f"[{scenario_label}] Run {j + 1}/{n_runs} done in {t1 - t0:.1f}s",
            flush=True,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """
    Parse CLI arguments, prepare recycling cost files, and dispatch workers.

    Returns:
    None
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run RTN scenarios in parallel for all_landfills and "
            "true_landfills across a set of recycling cost ratios."
        )
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        required=True,
        metavar="RATIO",
        help=(
            "Scale factors to apply to RecyclingCost_$ "
            "(e.g. 0.9 1.0 1.05). "
            "Use = notation for values < 1 to avoid shell flag ambiguity: "
            "--ratios=0.75 1.0"
        ),
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=_N_RUNS_DEFAULT,
        metavar="N",
        help=f"Model runs per scenario (default: {_N_RUNS_DEFAULT}).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=_N_STEPS_DEFAULT,
        metavar="N",
        help=f"Simulation length in years (default: {_N_STEPS_DEFAULT}).",
    )
    parser.add_argument(
        "--uspvdb-file",
        type=str,
        default=str(_USPVDB_FILE),
        metavar="FILE",
        help=f"Path to the USPVDB Excel file (default: {_USPVDB_FILE}).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum parallel worker processes. "
            "Defaults to min(scenarios, cpu_count)."
        ),
    )
    parser.add_argument(
        "--cost-component",
        type=str,
        choices=["recycling", "transport"],
        default="recycling",
        help=(
            "Which cost component to scale: 'recycling' scales RecyclingCost_$ "
            "in the recycling shipment files; 'transport' scales TransportCost_$ "
            "in both recycling and landfill shipment files. Default: recycling."
        ),
    )

    args = parser.parse_args()

    invalid: list[float] = [r for r in args.ratios if r <= 0]
    if invalid:
        parser.error(f"All ratios must be positive. Invalid: {invalid}")

    if not Path(args.uspvdb_file).is_file():
        parser.error(f"USPVDB file not found: {args.uspvdb_file}")

    # ── Phase 1: sequential file preparation ─────────────────────────────────
    scenarios: list[dict] = []
    for ratio in args.ratios:
        suffix: str = _build_suffix(ratio, args.cost_component)
        for set_name, config in _LANDFILL_SETS.items():
            print(
                f"\nPreparing: {set_name}, ratio={ratio} (suffix={suffix})"
            )
            recycling_filename, landfill_filename = _prepare_cost_files(
                set_config=config,
                ratio=ratio,
                suffix=suffix,
                uspvdb_file=args.uspvdb_file,
                cost_component=args.cost_component,
            )
            output_dir: str = str(
                _RESULTS_BASE / f"{config['results_prefix']}{suffix}"
            )
            scenarios.append(
                {
                    "recycling_filename": recycling_filename,
                    "landfill_filename": landfill_filename,
                    "output_dir": output_dir,
                    "label": f"{set_name}{suffix}",
                }
            )

    # ── Phase 2: parallel dispatch ────────────────────────────────────────────
    from utils import TIMESTEP as _TIMESTEP

    timestep_value: int = _TIMESTEP.QUARTERLY.value
    max_workers: int = args.max_workers or min(
        len(scenarios), os.cpu_count() or 1
    )

    print(
        f"\nDispatching {len(scenarios)} scenarios "
        f"({max_workers} parallel workers) ...\n"
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures: dict = {
            executor.submit(
                _run_scenario,
                s["recycling_filename"],
                s["landfill_filename"],
                s["output_dir"],
                args.n_runs,
                args.n_steps,
                timestep_value,
            ): s["label"]
            for s in scenarios
        }
        for future in as_completed(futures):
            label: str = futures[future]
            try:
                future.result()
                print(f"\n✓ Completed: {label}", flush=True)
            except Exception as exc:
                print(f"\n✗ Failed: {label} — {exc}", file=sys.stderr,
                      flush=True)


if __name__ == "__main__":
    main()
