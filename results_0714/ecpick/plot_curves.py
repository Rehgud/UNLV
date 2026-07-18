#!/usr/bin/env python3
"""러닝커브 CSV -> PNG (교수님 보고용).

ec_ecpick_pipeline.py 가 저장한 results/learncurve__<tag>.csv 를 읽어
  (1) train/val loss 곡선
  (2) 레벨별 val macro-F1 곡선
  (3) 레벨별 val micro-F1 곡선
을 하나의 PNG 로 그린다. 멤버(ensemble)별 평균 ± 범위 음영.

사용:
    python plot_curves.py                                  # results/ 안의 learncurve__* 전부
    python plot_curves.py results/learncurve__ECPICK__seq__train_ec__fixed.csv
헤드리스(디스플레이 없음) 환경에서도 동작(Agg 백엔드). matplotlib 필요.
"""
import glob
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

LEVELS = (1, 2, 3, 4)


def plot_one(csv_path):
    df = pd.read_csv(csv_path)
    g = df.groupby('epoch')
    ep = sorted(df['epoch'].unique())
    out = csv_path.replace('.csv', '.png')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(os.path.basename(csv_path).replace('learncurve__', '').replace('.csv', ''), fontsize=12)

    # (1) loss
    ax = axes[0]
    for col, color in [('train_loss', 'C0'), ('val_loss', 'C3')]:
        if col in df:
            m = g[col].mean(); lo = g[col].min(); hi = g[col].max()
            ax.plot(ep, m.values, color=color, label=col)
            ax.fill_between(ep, lo.values, hi.values, color=color, alpha=0.15)
    ax.set_title('Loss (train vs val)'); ax.set_xlabel('epoch'); ax.set_ylabel('loss')
    ax.legend(); ax.grid(alpha=0.3)

    # (2)(3) macro / micro F1
    for ax, kind in [(axes[1], 'macroF1'), (axes[2], 'microF1')]:
        for L in LEVELS:
            col = f'L{L}_{kind}'
            if col in df:
                m = g[col].mean(); lo = g[col].min(); hi = g[col].max()
                ax.plot(ep, m.values, label=f'L{L}')
                ax.fill_between(ep, lo.values, hi.values, alpha=0.12)
        ax.set_title(f'val {kind} (per level)'); ax.set_xlabel('epoch')
        ax.set_ylabel(kind); ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'saved -> {out}')


def main():
    args = sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    paths = args or sorted(glob.glob(os.path.join(here, 'results', 'learncurve__*.csv')))
    paths = [p for p in paths if not p.endswith('__mean.csv')]   # __mean 은 멤버평균(중복) → 개별만
    if not paths:
        print('learncurve CSV 없음. 먼저 학습을 돌리세요.')
        return
    for p in paths:
        plot_one(p)


if __name__ == '__main__':
    main()
