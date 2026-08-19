#!/usr/bin/env bash
#SBATCH --account=pvabm
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=absice_missing
#SBATCH --output=/projects/pvabm/pghosh/ABSiCE/torc_output/missing_scenario_%j.out
#SBATCH --mail-user=pghosh@nlr.gov
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

cd /projects/pvabm/pghosh/ABSiCE

module load conda
conda activate /projects/pvabm/pghosh/pv_abm

python hpc/run_single_scenario.py \
    --landfill-set true_landfills \
    --cost-rate 4.81 \
    --cost-component recycling \
    --rtn \
    --recycle-rate 0.60 \
    --n-runs 100 \
    --n-steps 11 \
    --results-base results/recycling_sensitivity
