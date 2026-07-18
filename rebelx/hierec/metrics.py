"""레벨별 P/R/F1 (macro & micro) — ECPICK 과 동일 지표로 공정 비교."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from hierarchy import LEVELS


def per_level_f1(P, Y, M, level, thr=0.5):
    """P:(n,N) MCM 확률, Y:(n,N) 정답, M:(n,4) 레벨마스크, level:(N,) 노드 레벨.

    각 레벨에서 '그 레벨을 아는 샘플'만, 그 레벨 노드들만 골라 지표 계산.
    """
    rows = []
    for L in LEVELS:
        cols = np.where(level == L)[0]
        mi = np.where(M[:, L - 1].astype(bool))[0]
        if len(mi) == 0 or len(cols) == 0:
            continue
        p = (P[np.ix_(mi, cols)] > thr).astype(int)
        y = Y[np.ix_(mi, cols)].astype(int)
        rows.append({'level': L, 'n_eval': int(len(mi)),
                     'macroP': precision_score(y, p, average='macro', zero_division=0),
                     'macroR': recall_score(y, p, average='macro', zero_division=0),
                     'macroF1': f1_score(y, p, average='macro', zero_division=0),
                     'microP': precision_score(y, p, average='micro', zero_division=0),
                     'microR': recall_score(y, p, average='micro', zero_division=0),
                     'microF1': f1_score(y, p, average='micro', zero_division=0)})
    return pd.DataFrame(rows)
