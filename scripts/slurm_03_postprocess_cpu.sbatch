#!/bin/bash
#SBATCH --job-name=nlp-postprocess
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---- Edit for your university cluster ----
# module load anaconda3
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate nlp-project
# or: source /path/to/venv/bin/activate

REPO_DIR="${REPO_DIR:-$HOME/nlp-group-project}"
cd "$REPO_DIR"
mkdir -p logs outputs

python -m pip install -r requirements.txt

python scripts/run_postprocessing.py \
  --tweets-emotions outputs/tweets_with_emotions.csv \
  --match-events outputs/match_events_cleaned.csv \
  --pressure-windows outputs/pressure_windows.csv \
  --output-dir outputs \
  --max-ads-per-match 4 \
  --min-gap-windows 2
