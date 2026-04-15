#!/bin/bash
#SBATCH --job-name=nlp-preprocess
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

python scripts/run_preprocessing.py \
  --tweet-path data/premier_league_twitter_comments_match_windows_2020_07_09_to_2020_10_13.csv \
  --match-path data/premier_league_combined_dataset_2020.csv \
  --output-dir outputs
