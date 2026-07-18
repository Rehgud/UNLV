#!/usr/bin/env python3
"""클래스별 성능(perclass__*.csv)을 '학습 개수 구간'별로 요약.

질문: 학습 예시가 몇 개 이상이면 잘 맞히고, 몇 개 이하면 못 맞히나?
-> train_count 를 구간(bin)으로 묶어 레벨별 recall/f1/정확히 맞춘 클래스 비율을 집계.

사용:
    python analyze_by_frequency.py                       # results/perclass__*.csv 자동
    python analyze_by_frequency.py results/perclass__....csv
결과: 콘솔 표 + results/freq_summary__<tag>.csv 저장
"""
import glob
import os
import sys
import numpy as np
import pandas as pd

BINS = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 500), (501, 10**9)]


def blabel(lo, hi):
    return f'{lo}-{hi}' if hi < 10**9 else f'{lo}+'


def summarize(df):
    out = []
    for L in sorted(df['level'].unique()):
        d = df[df['level'] == L]
        # test 에 등장한(정답이 1개 이상 있는) 클래스만 = 실제로 평가된 클래스
        d = d[d['test_support'] > 0]
        for lo, hi in BINS:
            b = d[(d['train_count'] >= lo) & (d['train_count'] <= hi)]
            if len(b) == 0:
                continue
            tp, fp, fn = b['TP'].sum(), b['FP'].sum(), b['FN'].sum()
            micro_r = tp / (tp + fn) if (tp + fn) else 0.0        # 이 구간 클래스들의 재현율(놓치지 않는 비율)
            micro_p = tp / (tp + fp) if (tp + fp) else 0.0
            macro_f1 = b['f1'].mean()                            # 클래스별 f1 평균
            solved = (b['recall'] > 0).mean()                    # 최소 1개라도 맞춘 클래스 비율
            out.append({'level': L, 'train_bin': blabel(lo, hi),
                        'n_classes': len(b), 'test_samples': int(b['test_support'].sum()),
                        'micro_recall': round(micro_r, 3), 'micro_prec': round(micro_p, 3),
                        'macro_f1': round(macro_f1, 3), 'classes_hit_ratio': round(solved, 3)})
    return pd.DataFrame(out)


def main():
    args = sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    paths = args or sorted(glob.glob(os.path.join(here, 'results', 'perclass__*.csv')))
    if not paths:
        print('perclass CSV 없음. 파이프라인(--eval fixed)을 먼저 돌리세요.')
        return
    for p in paths:
        df = pd.read_csv(p)
        summ = summarize(df)
        print(f'\n===== {os.path.basename(p)} =====')
        print('학습개수 구간별 성능 (test에 등장한 클래스만):')
        print(summ.to_string(index=False))
        outp = p.replace('perclass__', 'freq_summary__')
        summ.to_csv(outp, index=False)
        print(f'saved -> {outp}')
        # 레벨4 저빈도 요약 한 줄
        d4 = df[(df['level'] == 4) & (df['test_support'] > 0)]
        if len(d4):
            lo = d4[d4['train_count'] <= 10]; hi = d4[d4['train_count'] >= 101]
            print(f'  L4 요약: 학습 <=10개 클래스 recall {lo["recall"].mean():.3f} '
                  f'vs >=101개 클래스 recall {hi["recall"].mean():.3f}')


if __name__ == '__main__':
    main()
