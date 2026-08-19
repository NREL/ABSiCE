# -*- coding: utf-8 -*-
"""
RTN-optional scenario runner core for HPC calibration / recycling-sensitivity
runs.

This module builds one fully-resolved :class:`SimulationConfig` per scenario
on top of ``hpc/rtn_base.yaml``, writes it to a per-scenario YAML file, and
dispatches replicate runs through the existing :func:`absice.runner.run_batch`
process pool. It does not re-implement parallel execution, seeding, or CSV
writing — those responsibilities stay in ``absice/runner.py``.

The scenario runner works whether or not RTN is enabled
(``config.data_source.rtn``):

- RTN mode (``rtn=True``): the recycling-cost CSV is scaled and regenerated
  under ``RTN/`` via :mod:`hpc.cost_scaling`, and
  ``config.legacy_data.file_name["Recycling data"]`` is pointed at the
  scaled file. This requires the raw RTN shipment data (``RTN_Data``) and
  fails loudly if it is missing.
- Non-RTN mode (``rtn=False``): ``config.cost.original_recycling_cost`` is
  scaled directly. No RTN filesystem access is required.

Output layout mirrors the old kwargs-runner scheme (see
``landfill-paper-june-2026:hpc/run_single_scenario.py`` for reference, not
checked out on this branch):

    results_base/
      [att_mean_X.XX/]              # only when att_mean != base default
      recycle_rate_<XX>pct/
        <results_prefix><suffix>/
          Results_model_run_<id>.csv
          Results_agents_consumers_run_<id>.csv
          scenario_config.yaml      # resolved config used for this scenario

Transport-cost sensitivity is out of scope this session.
"""

from pathlib import Path

from hpc import cost_scaling
from absice.runner import run_batch
from absice.schemas.simulation_config import SimulationConfig
from utils import TIMESTEP

_HPC_DIR: Path = Path(__file__).parent.resolve()
_PROJECT_ROOT: Path = _HPC_DIR.parent
_BASE_CONFIG_PATH: Path = _HPC_DIR / "rtn_base.yaml"


def run_scenario(
    landfill_set: str,
    ratio: float,
    cost_component: str,
    n_runs: int,
    n_steps: int,
    results_base: Path,
    recycle_rate: float,
    att_mean: float,
    att_std: float,
    rtn: bool,
    workers: int | None = None,
    project_root: Path | None = None,
) -> list[Path]:
    """
    Build, write, and execute one calibration / sensitivity scenario.

    Parameters
    ----------
    landfill_set
        Key into :data:`hpc.cost_scaling.LANDFILL_SETS`
        (``"all_landfills"`` or ``"true_landfills"``). Selects the
        results-folder label prefix in every mode; in RTN mode it also
        selects which RTN shipment file and recycling-cost CSV prefix to
        use.
    ratio
        Recycling-cost scale multiplier (e.g. 1.0 for baseline, 1.25 for
        +25%). In RTN mode this scales ``RecyclingCost_$`` in the shipment
        data; in non-RTN mode it scales
        ``config.cost.original_recycling_cost``. Convert an absolute
        $/kg cost rate to a ratio with
        :func:`hpc.cost_scaling.ratio_from_cost_rate` before calling this
        function (RTN-specific; do not reuse that conversion for
        non-RTN mode).
    cost_component
        Cost column to scale. Only ``"recycling"`` is supported this
        session; transport sensitivity is out of scope.
    n_runs
        Number of independent replicate simulations.
    n_steps
        Simulation length in years (converted to quarterly steps
        internally: ``n_steps * 4``).
    results_base
        Base directory under which the scenario's output subdirectory
        tree is created.
    recycle_rate
        Initial recycling EoL rate (0.0-1.0). The delta versus the base
        config's own baseline recycle rate is subtracted from the landfill
        rate so all EoL rates continue to sum to 1.
    att_mean
        Mean of the attitude-toward-recycling distribution
        (``tpb.att_distrib_param_eol[0]``). An ``att_mean_X.XX/``
        subdirectory is inserted into the output path only when this
        differs from the base config's own default value.
    att_std
        Standard deviation of the attitude distribution
        (``tpb.att_distrib_param_eol[1]``).
    rtn
        Whether to run in RTN mode (``config.data_source.rtn``). Non-RTN
        mode requires no RTN_Data filesystem access.
    workers
        Number of parallel worker processes for ``run_batch``. Defaults to
        the available CPU count.
    project_root
        Root directory of the ABSiCE repository. Defaults to the parent of
        this file's directory.

    Returns
    -------
    list[Path]
        Paths to all output CSV files, ordered by run ID (as returned by
        :func:`absice.runner.run_batch`).

    Raises
    ------
    KeyError
        If ``landfill_set`` is not a recognized key.
    ValueError
        If ``cost_component`` is not ``"recycling"``.
    FileNotFoundError
        If ``rtn=True`` and the required RTN_Data or USPVDB base files are
        missing.
    AssertionError
        If the rebalanced ``init_eol_rate`` does not sum to 1.
    """
    if cost_component != "recycling":
        raise ValueError(
            "Only cost_component='recycling' is supported this session; "
            "transport sensitivity is out of scope."
        )

    if landfill_set not in cost_scaling.LANDFILL_SETS:
        raise KeyError(
            f"Unknown landfill_set '{landfill_set}'. "
            f"Choose from: {list(cost_scaling.LANDFILL_SETS.keys())}"
        )

    project_root = (
        Path(project_root).resolve()
        if project_root is not None
        else _PROJECT_ROOT
    )
    results_base = Path(results_base).resolve()

    # ── 1. Load the pinned published-run base config ─────────────────────
    config = SimulationConfig.from_yaml(_BASE_CONFIG_PATH)

    # Capture base-config baselines *before* mutating, so the rebalancing
    # and att_mean-subdirectory logic stay config-driven instead of relying
    # on hardcoded magic numbers.
    base_init_eol_rate = dict(config.eol.init_eol_rate)
    base_att_mean = config.tpb.att_distrib_param_eol[0]

    # ── 2. Apply per-scenario overrides ───────────────────────────────────
    config.data_source.rtn = rtn

    config.run.timestep = TIMESTEP.QUARTERLY
    config.run.last_step = n_steps * 4

    baseline_recycle = base_init_eol_rate["recycle"]
    landfill_rate = base_init_eol_rate["landfill"] - (
        recycle_rate - baseline_recycle
    )
    new_init_eol_rate = {
        **base_init_eol_rate,
        "recycle": recycle_rate,
        "landfill": round(landfill_rate, 10),
    }
    assert abs(sum(new_init_eol_rate.values()) - 1.0) < 1e-9, (
        f"init_eol_rate does not sum to 1: {new_init_eol_rate}"
    )
    # The sum==1 assertion above cannot catch an out-of-range individual
    # rate (e.g. a large --recycle-rate driving "landfill" negative while
    # some other pathway compensates to keep the total at 1). Guard every
    # rate explicitly (R2).
    out_of_range = {
        pathway: rate
        for pathway, rate in new_init_eol_rate.items()
        if not (0.0 <= rate <= 1.0)
    }
    if out_of_range:
        raise ValueError(
            "init_eol_rate has pathway(s) outside [0, 1] after rebalancing "
            f"recycle_rate={recycle_rate}: {out_of_range}. "
            f"Full init_eol_rate: {new_init_eol_rate}"
        )
    config.eol.init_eol_rate = new_init_eol_rate

    config.tpb.att_distrib_param_eol = [att_mean, att_std]

    suffix = cost_scaling.build_suffix(ratio, cost_component)

    if rtn:
        recycling_filename = cost_scaling.prepare_rtn_recycling_cost_file(
            landfill_set=landfill_set,
            ratio=ratio,
            suffix=suffix,
        )
        config.legacy_data.file_name.recycling_data = recycling_filename
        # Landfill-cost scaling is out of scope this session, so this is
        # always the unscaled base RTN landfill CSV for the requested
        # landfill_set; it only needs to be set (not regenerated) so the
        # model's RTN landfill-cost load (ABM_CE_PV_Model.py, `if self.rtn`)
        # finds the file matching the scenario's landfill_set.
        config.legacy_data.file_name.rtn_landfill_data = (
            cost_scaling.LANDFILL_SETS[landfill_set]["landfill_filename"]
        )
    else:
        config.cost.original_recycling_cost = (
            cost_scaling.scale_original_recycling_cost(
                config.cost.original_recycling_cost, ratio,
            )
        )

    # ── 3. Build the scenario output label (old kwargs-runner scheme) ────
    set_config = cost_scaling.LANDFILL_SETS[landfill_set]
    recycle_rate_label = f"recycle_rate_{int(round(recycle_rate * 100))}pct"

    if abs(att_mean - base_att_mean) > 1e-9:
        label = (
            f"att_mean_{att_mean:.2f}/{recycle_rate_label}/"
            f"{set_config['results_prefix']}{suffix}"
        )
    else:
        label = (
            f"{recycle_rate_label}/{set_config['results_prefix']}{suffix}"
        )

    # ── 4. Write the resolved config to a per-scenario YAML ──────────────
    scenario_dir = results_base / label
    scenario_dir.mkdir(parents=True, exist_ok=True)
    scenario_config_path = scenario_dir / "scenario_config.yaml"
    config.to_yaml(scenario_config_path)

    # ── 5. Dispatch replicate runs through the shared process pool ───────
    return run_batch(
        config_path=scenario_config_path,
        paths_yaml=None,
        n_runs=n_runs,
        workers=workers,
        output_dir=results_base,
        label=label,
        project_root=project_root,
    )
