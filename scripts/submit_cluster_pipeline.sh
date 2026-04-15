#!/bin/bash
set -euo pipefail

# Submit all 3 jobs with dependencies: preprocess -> emotion(gpu) -> postprocess
PRE_JOB=$(sbatch scripts/slurm_01_preprocess_cpu.sbatch | awk '{print $4}')
GPU_JOB=$(sbatch --dependency=afterok:${PRE_JOB} scripts/slurm_02_emotion_gpu.sbatch | awk '{print $4}')
POST_JOB=$(sbatch --dependency=afterok:${GPU_JOB} scripts/slurm_03_postprocess_cpu.sbatch | awk '{print $4}')

echo "Submitted preprocess job: ${PRE_JOB}"
echo "Submitted emotion GPU job: ${GPU_JOB}"
echo "Submitted postprocess job: ${POST_JOB}"
