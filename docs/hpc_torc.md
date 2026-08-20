# Running ABSiCE on HPC with TORC

Run calibration/sensitivity sweeps on a Slurm cluster using
[TORC](https://github.com/NatLabRockies/torc). Each sweep expands into one job
per parameter combination; each job runs `--n-runs` replicates.

## Prerequisites

- Slurm account on the cluster.
- The `pv_abm` conda environment installed (see the main [README](../README.md#installation)).
- The `torc` client and `torc-server` binary available (`torc --version` works).
- RTN runs only: `RTN_Data/Round_3` plus the cost CSVs under `RTN/`. Non-RTN
  runs need none of this.

## 1. Fill in placeholders

The scripts use placeholders instead of personal values. Replace them before use:

| Placeholder | Meaning | Files |
| --- | --- | --- |
| `<YOUR_EMAIL>` | Slurm mail address | `hpc/torc_*.yaml`, `hpc/run_missing_scenario.sh` |
| `<CONDA_ENV_PATH>` | Path to the `pv_abm` conda env | `hpc/torc_*.yaml`, `hpc/run_missing_scenario.sh` |
| `<PROJECT_DIR>` | ABSiCE checkout path on the cluster | `hpc/run_missing_scenario.sh` |
| `<TORC_BINARY_DIR>` | Dir containing `torc-server` | `hpc/start_torc_server.sh` |
| `<HPC_RTN_DATA_DIR>` / `<LOCAL_RTN_DATA_DIR>` | Parent of `RTN_Data/Round_3` (RTN only) | `hpc/cost_scaling.py` |

Also set `account`/`partition`/`nodes`/`walltime` in the `torc_*.yaml` specs for
your cluster.

## 2. Start the TORC server

`submit.sh` reaches the server via `TORC_API_URL`. **Source** the script so the
export persists in your shell:

```bash
source hpc/start_torc_server.sh                 # OS-assigned port, current node
source hpc/start_torc_server.sh --host <login-node> --port 52619
```

Pass `--host` as the login node you are on (run `hostname` to find it); it
defaults to the current node's FQDN if omitted.

Verify: `echo "$TORC_API_URL"` is non-empty. (With a shared server, just
`export TORC_API_URL=http://<host>:<port>/torc-service/v1`.)

## 3. Submit a sweep

```bash
TORC_ACCOUNT=<account> ./hpc/submit.sh recycling
TORC_ACCOUNT=<account> ./hpc/submit.sh att_calibration
```

Logs go to `torc_output/<run_name>/`. Edit the `parameters:` lists in a spec to
change the sweep.

## Local smoke test (no cluster/server)

```bash
python hpc/run_single_scenario.py --ratio 1.0 --rtn false \
    --n-runs 1 --n-steps 1 --results-base /tmp/smoke
```

## Scenario flags

`hpc/run_single_scenario.py` (`--help` for all):

| Flag | Purpose |
| --- | --- |
| `--rtn {true,false}` | RTN or non-RTN mode (default `false`). |
| `--landfill-set {all_landfills,true_landfills}` | Required in RTN mode; ignored otherwise. |
| `--ratio FLOAT` \| `--cost-rate RATE` | Recycling-cost scale (multiplier, or $/kg). Both modes. |
| `--recycle-rate RATE` | Initial recycle EoL rate (rebalanced to sum to 1). |
| `--att-mean` / `--att-std` | Attitude distribution parameters. |
| `--n-runs` / `--n-steps` | Replicates / simulation years. |
| `--results-base DIR` | Output base directory. |

Base config: `hpc/rtn_base.yaml`. **RTN** (`--rtn true`) reads cost CSVs from
`RTN/` and requires `RTN_Data`; **non-RTN** scales
`config.cost.original_recycling_cost` directly (no RTN data).

## Output

```
<results-base>/[att_mean_X.XX/]recycle_rate_XXpct/<label><suffix>/
    Results_model_run_<i>.csv
    Results_agents_consumers_run_<i>.csv
    scenario_config.yaml
```

`<label>` is `RTN_run_<landfill_set>` (RTN) or `run` (non-RTN).

Analyse a calibration sweep:

```bash
python hpc/analyze_att_calibration.py --results-base results/att_calibration
```
