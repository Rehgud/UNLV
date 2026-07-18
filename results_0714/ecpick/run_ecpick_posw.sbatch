#!/bin/bash
# ECPICK 백업 실행 — 기존 설정(pos_weight ON, threshold 0.5).
#   지금 도는 OFF/0.19 실행과 병렬. 출력은 results_posw/ 로 분리(충돌 방지).
#SBATCH --job-name=ecpick-posw
#SBATCH --partition=gpuq-a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=ecpick_posw_%j.log

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ecpick
cd "${SLURM_SUBMIT_DIR:-$HOME/DOHYEONG/rebelx/ecpick}"

echo "[node] $(hostname)  [gpu] ${CUDA_VISIBLE_DEVICES:-?}  [pwd] $(pwd)  [MODE] pos_weight ON / thr 0.5"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u ec_ecpick_pipeline.py \
    --eval fixed \
    --train-csv ../data/train_ec.csv \
    --test-csv  ../data/test_ec.csv \
    --results-dir results_posw \
    --ensemble 10 \
    --batch 32 \
    --epochs 20 \
    --num-workers 4 \
    --val-every 1 \
    --seed 0 \
    --class-weights 1 \
    --threshold 0.5

echo "=== ECPICK(pos_weight ON) 완료 -> results_posw/ECPICK__seq__train_ec.csv ==="
