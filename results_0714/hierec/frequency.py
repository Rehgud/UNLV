"""HierEC 클래스별 성능 + 학습개수 구간(빈도) 요약.

ECPICK 의 analyze_by_frequency 와 '동일한 스키마'로 CSV 를 저장해, 두 모델을
같은 방식으로 비교/그래프화할 수 있게 한다.
  - perclass__hier_chmcnn__{tag}.csv : level, ec_class, train_count, test_support, TP, FP, FN, precision, recall, f1
  - freq_summary__hier_chmcnn__{tag}.csv : level, train_bin, n_classes, test_samples,
                                           micro_recall, micro_prec, macro_f1, classes_hit_ratio
"""
import numpy as np
import pandas as pd
from hierarchy import LEVELS

# ECPICK freq_summary 와 동일한 학습개수 구간
BINS = [(1, 2, '1-2'), (3, 5, '3-5'), (6, 10, '6-10'), (11, 20, '11-20'),
        (21, 50, '21-50'), (51, 100, '51-100'), (101, 500, '101-500'), (501, 10**9, '501+')]


def _bin(c):
    for lo, hi, name in BINS:
        if lo <= c <= hi:
            return name
    return None


def per_class_and_freq(P, Yte, Mte, level, nodes, train_count, thr=0.5):
    """MCM 확률 P(n,N) 로 노드(=EC 클래스)별 TP/FP/FN 및 빈도구간 요약 계산.

    per_level_f1 과 동일한 마스킹: 레벨 L 노드는 '그 레벨을 아는 test 샘플'만으로 평가.
    """
    pred = (P > thr)
    rows = []
    for j, name in enumerate(nodes):
        L = int(level[j])
        mi = np.where(Mte[:, L - 1].astype(bool))[0]      # 레벨 L 을 아는 샘플만
        if len(mi) == 0:
            continue
        yj = Yte[mi, j].astype(bool)
        pj = pred[mi, j]
        TP = int((pj & yj).sum())
        FP = int((pj & ~yj).sum())
        FN = int((~pj & yj).sum())
        prec = TP / (TP + FP) if TP + FP > 0 else 0.0
        rec = TP / (TP + FN) if TP + FN > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        rows.append({'level': L, 'ec_class': name, 'train_count': int(train_count[j]),
                     'test_support': int(yj.sum()), 'TP': TP, 'FP': FP, 'FN': FN,
                     'precision': round(prec, 6), 'recall': round(rec, 6), 'f1': round(f1, 6)})
    perclass = pd.DataFrame(rows)

    # 빈도 요약 — test 에 등장한 클래스(test_support>0)만, ECPICK 방식과 동일
    d = perclass[perclass.test_support > 0].copy()
    d['train_bin'] = d['train_count'].apply(_bin)
    summ = []
    for L in LEVELS:
        for lo, hi, name in BINS:
            g = d[(d.level == L) & (d.train_bin == name)]
            if len(g) == 0:
                continue
            TP, FP, FN = g.TP.sum(), g.FP.sum(), g.FN.sum()
            micro_rec = TP / (TP + FN) if TP + FN > 0 else 0.0
            micro_prec = TP / (TP + FP) if TP + FP > 0 else 0.0
            summ.append({'level': L, 'train_bin': name, 'n_classes': int(len(g)),
                         'test_samples': int(g.test_support.sum()),
                         'micro_recall': round(micro_rec, 3), 'micro_prec': round(micro_prec, 3),
                         'macro_f1': round(g.f1.mean(), 3),
                         'classes_hit_ratio': round(float((g.recall > 0).mean()), 3)})
    return perclass, pd.DataFrame(summ)
