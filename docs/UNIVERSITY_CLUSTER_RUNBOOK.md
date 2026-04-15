# University GPU Cluster Runbook (Current Repo Setup)

This runbook is tailored to this repository and the scripts in `scripts/`.

## 0. What runs on GPU vs CPU

- CPU job 1: preprocessing (`scripts/run_preprocessing.py`)
- GPU job: transformer emotion inference (`scripts/run_emotion_inference.py`)
- CPU job 2: alignment + scoring + evaluation tables (`scripts/run_postprocessing.py`)

This is the efficient split because only transformer inference is GPU-bound.

## 1. One-time local checks

From your laptop:

```bash
cd /Users/sofiiaavetisian/Desktop/UNI/NLP/group_project/nlp-group-project
ls scripts
```

You should see:
- `run_preprocessing.py`
- `run_emotion_inference.py`
- `run_postprocessing.py`
- `slurm_01_preprocess_cpu.sbatch`
- `slurm_02_emotion_gpu.sbatch`
- `slurm_03_postprocess_cpu.sbatch`
- `submit_cluster_pipeline.sh`

## 2. Copy repo to cluster

Option A (recommended): push to GitHub and clone on cluster.

Option B (direct copy with rsync):

```bash
rsync -av --progress \
  /Users/sofiiaavetisian/Desktop/UNI/NLP/group_project/nlp-group-project/ \
  <your_user>@<cluster_login_host>:~/nlp-group-project/
```

## 3. SSH into cluster

```bash
ssh <your_user>@<cluster_login_host>
cd ~/nlp-group-project
```

## 4. Create/activate environment on cluster

Use your university's preferred environment manager.

### Conda example

```bash
module load anaconda3
conda create -y -n nlp-project python=3.10
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nlp-project
pip install -r requirements.txt
```

### venv example

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Edit Slurm templates for your cluster

Open and edit these files:
- `scripts/slurm_01_preprocess_cpu.sbatch`
- `scripts/slurm_02_emotion_gpu.sbatch`
- `scripts/slurm_03_postprocess_cpu.sbatch`

Fields you usually must change:
- `#SBATCH --partition=...`
- Add account/QOS if your uni requires it (for example `#SBATCH --account=<account_name>`)
- Module/conda activation lines
- `REPO_DIR` default if your repo path is not `~/nlp-group-project`

If your cluster does not use Slurm, stop here and convert these to your scheduler format (PBS/LSF).

## 6. Validate scripts before submit

```bash
bash -n scripts/slurm_01_preprocess_cpu.sbatch
bash -n scripts/slurm_02_emotion_gpu.sbatch
bash -n scripts/slurm_03_postprocess_cpu.sbatch
python scripts/run_preprocessing.py --help
python scripts/run_emotion_inference.py --help
python scripts/run_postprocessing.py --help
```

## 7. Submit jobs (recommended: dependency chain)

```bash
chmod +x scripts/submit_cluster_pipeline.sh
./scripts/submit_cluster_pipeline.sh
```

This submits:
1. preprocess CPU
2. emotion GPU (runs only if preprocess succeeded)
3. postprocess CPU (runs only if GPU job succeeded)

## 8. Monitor jobs

```bash
squeue -u $USER
sacct -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS
```

Inspect logs:

```bash
ls -lt logs | head
tail -n 120 logs/nlp-preprocess-<jobid>.out
tail -n 120 logs/nlp-emotion-<jobid>.out
tail -n 120 logs/nlp-postprocess-<jobid>.out
```

## 9. Expected outputs

After all jobs complete, these files should exist:

Preprocess outputs:
- `outputs/tweets_cleaned.csv`
- `outputs/match_events_cleaned.csv`
- `outputs/pressure_windows.csv`

GPU inference output:
- `outputs/tweets_with_emotions.csv`

Postprocess outputs:
- `outputs/aligned_windows.csv`
- `outputs/scored_windows.csv`
- `outputs/recommended_ad_slots.csv`
- `outputs/match_policy_summary.csv`
- `outputs/correlation_matrix.csv`
- `outputs/window_stats.csv`

## 10. Download outputs back to laptop

Run on your laptop:

```bash
rsync -av --progress \
  <your_user>@<cluster_login_host>:~/nlp-group-project/outputs/ \
  /Users/sofiiaavetisian/Desktop/UNI/NLP/group_project/nlp-group-project/outputs/
```

## 11. Troubleshooting

### A) `sbatch: error: Invalid partition`
Use your real partition names from cluster docs:
- CPU partition for preprocess/postprocess
- GPU partition for emotion job

### B) GPU job starts but fails with CUDA OOM
Lower batch size in `scripts/slurm_02_emotion_gpu.sbatch`:

```bash
--batch-size 128
```

If still failing, try `64`.

### C) `ModuleNotFoundError`
Environment activation is wrong in sbatch script. Fix the module/conda/venv lines and resubmit.

### D) `Input file not found` in GPU or postprocess jobs
Dependency chain likely broken or job submitted manually out of order.
Run in order: preprocess -> emotion -> postprocess.

### E) Long queue wait for GPU
You can run preprocess and postprocess quickly on CPU while waiting; only stage 2 needs GPU.

## 12. Manual submission commands (if you don’t want helper script)

```bash
PRE=$(sbatch scripts/slurm_01_preprocess_cpu.sbatch | awk '{print $4}')
GPU=$(sbatch --dependency=afterok:$PRE scripts/slurm_02_emotion_gpu.sbatch | awk '{print $4}')
POST=$(sbatch --dependency=afterok:$GPU scripts/slurm_03_postprocess_cpu.sbatch | awk '{print $4}')
echo "$PRE $GPU $POST"
```

## 13. Quick single-stage reruns

Rerun only GPU stage:

```bash
sbatch scripts/slurm_02_emotion_gpu.sbatch
```

Rerun only postprocess stage:

```bash
sbatch scripts/slurm_03_postprocess_cpu.sbatch
```

## 14. What to write in your methodology section

"We used the university GPU cluster only for transformer-based emotion inference (GoEmotions RoBERTa), the computational bottleneck. Preprocessing and downstream window scoring were executed on CPU nodes to minimize GPU queue usage and improve resource efficiency."
