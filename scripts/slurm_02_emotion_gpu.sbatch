#!/bin/bash
#SBATCH --job-name=nlp-emotion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---- Edit for your university cluster ----
# module load anaconda3 cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate nlp-project
# or: source /path/to/venv/bin/activate

REPO_DIR="${REPO_DIR:-$HOME/nlp-group-project}"
cd "$REPO_DIR"
mkdir -p logs outputs

python -m pip install -r requirements.txt

python scripts/run_emotion_inference.py \
  --input outputs/tweets_cleaned.csv \
  --output outputs/tweets_with_emotions.csv \
  --batch-size 256
