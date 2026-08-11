# ABSiCE Quick-Start Guide

## 1. Clone and set up the environment

```bash
git clone https://github.com/NatLabRockies/ABSiCE.git 
cd ABSiCE
conda env create -f pv_abm_env_platform_independent.yaml
conda activate pv_abm
```

## 2. Check that the required data files are in place

The simulation reads pre-processed data files that are not stored in the repo
(they are too large). Ask a team member for the `TEMP/` and `PV_ICE/` directories
and place them under `ABSiCE/`.

To verify all paths resolve correctly before running:
```bash
python -c "
from pathlib import Path
from absice.schemas.data_paths_config import DataPathsConfig
paths = DataPathsConfig.make_default(Path('.'))
print('All paths configured — check that the files exist before running.')
"
```

## 3. Configure your scenario

Copy the default configuration and edit it:
```bash
python run.py init-config --output config/my_scenario.yaml
# Open config/my_scenario.yaml in any text editor and change what you need.
```

Common things to change:
- `run.seed` — set an integer for reproducible results (e.g. `42`)
- `consumer.model_states` — restrict to a subset of US states (e.g. `[California, Texas]`)
- `scenario.hazardous_waste_regulation_enabled` — turn the regulation on/off

## 4. Run a single simulation

```bash
python run.py run --config config/my_scenario.yaml
```

Output is written to `results/my_scenario/Results_model_run_0.csv`.
The sub-folder name defaults to the config file stem so results from
different scenarios never overwrite each other.

## 5. Run multiple replicates in parallel

```bash
python run.py batch --config config/my_scenario.yaml --n-runs 10 --workers 4
```

This launches 10 independent runs (seeds 0–9) across 4 CPU cores.
Each run writes its own CSV under `results/my_scenario/`:
`Results_model_run_0.csv` through `Results_model_run_9.csv`.

To run a second scenario without touching the first:
```bash
python run.py batch --config config/epr_scenario.yaml --n-runs 10 --workers 4
# writes to results/epr_scenario/ — completely separate from results/my_scenario/
```

You can also set the label explicitly:
```bash
python run.py batch --config config/my_scenario.yaml --label baseline_v2 --n-runs 10
# writes to results/baseline_v2/
```

## 6. Compare runs

Use the existing `compare_results.py` script to diff outputs:
```bash
python compare_results.py results/Results_model_run_0.csv results/Results_model_run_1.csv
```