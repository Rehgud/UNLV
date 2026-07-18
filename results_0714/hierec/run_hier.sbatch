#!/bin/bash
# C-HMCNN (계층 인식 head) — Slurm job
#   제출: sbatch run_hier.sbatch
#   확인: squeue -u sp00002  /  tail -f hier_<jobid>.log
#SBATCH --job-name=hier-ec
#SBATCH --partition=gpuq-a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=hier_%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ecpick
cd "${SLURM_SUBMIT_DIR:-$HOME/DOHYEONG/rebelx/hierec}"

# ─────────────────────────────────────────────────────────────────────────
#  실제 ESM-2 3B feature 로 학습 (features_all.pt: ids/features[N,2560]/targets/labels).
#  라벨은 features_all.pt 안에 들어있어 별도 CSV 불필요. split.json = {train,valid,test} id목록.
#  경로는 홈에 있는 파일 기준 — 다른 곳이면 --features/--split 만 바꿀 것.
# ─────────────────────────────────────────────────────────────────────────
FEAT="${FEAT:-$HOME/features_all.pt}"
SPLIT="${SPLIT:-$HOME/split.json}"

python -u train.py --features "$FEAT" --split "$SPLIT" \
    --epochs 300 --eval-every 10 --tag real --out results

echo "=== HierEC 완료 -> results/hier_chmcnn__real.csv ==="

# (배관만 확인하려면) 랜덤 feature 스모크:
# python -u train.py --synthetic --epochs 100 --eval-every 20
