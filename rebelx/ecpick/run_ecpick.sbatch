#!/bin/bash
# ECPICK — 팀원 전처리 데이터(고정 분할) 학습/평가 — REBELX Slurm job
#   제출:  sbatch run_ecpick.sbatch
#   확인:  squeue -u $USER   /   tail -f ecpick_<jobid>.log
#   결과:  results/ECPICK__seq__train_ec.csv  (+ 콘솔에 per-level F1)
#
#   ※ ECPICK 논문 설정과 동일:
#      ensemble=10, batch=32, dropout=0.8, relu=384, beta=0.6, class-weights on
#   ※ 평가: 고정 분할(train_ec 학습 -> test_ec 평가). val_ec 는 별도 보관.
#SBATCH --job-name=ecpick
#SBATCH --partition=gpuq-a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=ecpick_%j.log

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ecpick

# Slurm 은 스크립트를 spool 로 복사 → $0 폴더가 아님. 제출 폴더(=ecpick/)로 이동.
cd "${SLURM_SUBMIT_DIR:-$HOME/DOHYEONG/rebelx/ecpick}"

echo "[node] $(hostname)  [gpu] ${CUDA_VISIBLE_DEVICES:-?}  [pwd] $(pwd)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u ec_ecpick_pipeline.py \
    --eval fixed \
    --train-csv ../data/train_ec.csv \
    --test-csv  ../data/test_ec.csv \
    --results-dir results \
    --ensemble 10 \
    --batch 32 \
    --epochs 20 \
    --num-workers 4 \
    --val-every 1 \
    --seed 0 \
    --class-weights 0 \
    --threshold 0.19

echo "=== ECPICK 완료 -> results/ECPICK__seq__train_ec.csv ==="
