# -*- coding: utf-8 -*-
"""
Run parameter sweep calibration simulations without RTN or cost scaling.

This script performs calibration sweeps over attitude distribution parameters
(att_mean, att_std) without using the RTN flag or scaling any costs.
All costs and recycling rates are constant and set directly in ABM_CE_PV_Model.py.

Pipeline per parameter combination:
  1. No cost file preparation (costs are constant in the model)
  2. Run N model runs with specified parameter values
  3. Save results to results/calibration/...

Usage (local smoke test):
    python run_calibration_sweep.py \
        --att-means 0.5 0.515 0.53 \
        --n-runs 2 \
        --n-steps 2

Usage (full calibration):
    python run_calibration_sweep.py \
        --att-means 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 \
        --n-runs 100 \
        --n-steps 11
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
_RESULTS_BASE: Path = _WORKSPACE_DIR / "results" / "calibration"

# Calibrated run parameters — no RTN, constant costs
_N_RUNS_DEFAULT: int = 10
_N_STEPS_DEFAULT: int = 11  # simulation years; multiplied by timestep below

# Fixed model parameters — no RTN, no cost scaling
_FIXED_MODEL_PARAMS: dict = {
    "hazardous_waste_regulation_enabled": False,
    "landfill_solar_waste_acceptance_ratio": 1.0,
    "calculate_distances": False,
    "model_states": ["TX", "AZ", "NV", "NM"],
    "solar_cycle": True,
    "rtn": False,  # ← No RTN for calibration
}

# Default cost files (constant, not scaled)
_DEFAULT_RECYCLING_FILE: str = "Recyclers_data.csv"
_DEFAULT_LANDFILL_FILE: str = "Landfills_data_2023.csv"
_HAZARDOUS_LANDFILL_FILE: str = "Landfills_data_SA.csv"

# Default baseline EoL rates
_DEFAULT_INIT_EOL_RATE: dict = {
    "repair": 0.005, "sell": 0.01, "recycle": 0.2, "landfill": 0.785, "hoard": 0.0
}


# Default attitude std — matches calibration (Saphores 2012)
_ATT_STD_DEFAULT: float = 0.1


def _run_scenario(
    output_dir: str,
    n_runs: int,
    n_steps_years: int,
    timestep_value: int,
    att_mean: float,
) -> None:
    """
    Run n_runs model simulations for one parameter combination (no cost scaling).

    Saves Results_model_run_{j}.csv and Results_agents_consumers_run_{j}.csv
    per run. Uses constant costs (not scaled) as defined in the model.
    Recycling rates and attitude std dev are also constant as set in the model.

    Parameters:
    output_dir (str): Absolute path to the results subfolder.
    n_runs (int): Number of model runs to perform.
    n_steps_years (int): Simulation length in years.
    timestep_value (int): TIMESTEP enum value (e.g. 4 for QUARTERLY).
    att_mean (float): Mean of the bounded-normal attitude distribution.

    Returns:
    None
    """
    att_std = _ATT_STD_DEFAULT
    init_eol_rate = _DEFAULT_INIT_EOL_RATE

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

    # All-or-nothing skip: if every expected output file is already present,
    # the scenario is complete — skip it entirely.
    completed_runs: int = sum(
        1 for j in range(n_runs)
        if (output_path / f"Results_model_run_{j}.csv").exists()
        and (output_path / f"Results_agents_consumers_run_{j}.csv").exists()
    )
    if completed_runs == n_runs:
        print(
            f"[{scenario_label}] All {n_runs} runs already complete — skipping.",
            flush=True,
        )
        return

    if completed_runs > 0:
        print(
            f"[{scenario_label}] {completed_runs}/{n_runs} runs found but incomplete"
            f" — re-running full suite.",
            flush=True,
        )

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
            solar_cycle=True,
            filter_landfills_not_accepting_pv=True,
            rtn=False,  # No RTN — use model defaults
            init_eol_rate=init_eol_rate,
            att_distrib_param_eol=[att_mean, att_std],
            landfill_data_params = {
                    "landfill_volume_column": "Waste Business Journal Costs ($/metric tons)",
                    "landfill_name_column": "Landfill Name"
            },
            # file_name={
            #     "Landfill data": _DEFAULT_LANDFILL_FILE,
            #     "Recycling data": _DEFAULT_RECYCLING_FILE,
            #     "Hazardous landfill data": _HAZARDOUS_LANDFILL_FILE,
            # },
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
    Parse CLI arguments and dispatch parameter sweep workers.

    Returns:
    None
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run calibration parameter sweeps (no RTN, constant costs). "
            "Sweeps over attitude distribution parameters and recycling rates."
        )
    )
    parser.add_argument(
        "--att-means",
        nargs="+",
        type=float,
        required=True,
        metavar="MEAN",
        help=(
            "Attitude mean values to sweep "
            "(e.g. 0.50 0.55 0.60). "
            "Must be in [0, 1]. "
            f"Att std dev is fixed at {_ATT_STD_DEFAULT}."
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
        "--results-base",
        type=Path,
        default=_RESULTS_BASE,
        metavar="DIR",
        help=(
            "Base directory for scenario output folders "
            f"(default: {_RESULTS_BASE}). "
        ),
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

    args = parser.parse_args()

    # Validate attitude parameters
    invalid_means: list[float] = [
        m for m in args.att_means if m < 0 or m > 1
    ]
    if invalid_means:
        parser.error(f"All att_means must be in [0, 1]. Invalid: {invalid_means}")

    # ── Build scenario list ──────────────────────────────────────────────────
    scenarios: list[dict] = []
    results_base: Path = Path(args.results_base).resolve()

    for att_mean in args.att_means:
        att_label: str = f"att_mean_{att_mean:.3f}"
        output_dir: str = str(results_base / att_label)
        scenarios.append(
            {
                "output_dir": output_dir,
                "att_mean": att_mean,
                "label": f"att_mean={att_mean:.3f}",
            }
        )
    # ── Dispatch workers ─────────────────────────────────────────────────────
    from utils import TIMESTEP as _TIMESTEP

    timestep_value: int = _TIMESTEP.QUARTERLY.value
    max_workers: int = args.max_workers or min(len(scenarios), os.cpu_count() or 1)

    results_base.mkdir(parents=True, exist_ok=True)

    print(
        f"\nCalibration sweep: {len(scenarios)} scenarios "
        f"({max_workers} parallel workers) ...\n"
    )
    for s in scenarios:
        print(f"  • {s['label']}")
    print()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures: dict = {
            executor.submit(
                _run_scenario,
                s["output_dir"],
                args.n_runs,
                args.n_steps,
                timestep_value,
                s["att_mean"],
            ): s["label"]
            for s in scenarios
        }
        for future in as_completed(futures):
            label: str = futures[future]
            try:
                future.result()
                print(f"\n✓ Completed: {label}", flush=True)
            except Exception as exc:
                print(
                    f"\n✗ Failed: {label} — {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    print(f"\n=== Calibration sweep complete ===\n", flush=True)
    print(f"Results saved to: {results_base}\n", flush=True)


if __name__ == "__main__":
    main()
