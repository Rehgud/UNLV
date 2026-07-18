"""EC 계층 트리 구성 + C-HMCNN 용 조상/자손 관계.

노드 = 부분 EC 문자열 ('1', '1.1', '1.1.1', '1.1.1.1'). train 에 등장한 것만.
MCM(Max Constraint Module) 은 "부모 확률 = 자기+자손 확률의 max" 를 강제하는데,
EC 는 깊이가 최대 4 라서 (노드,조상) 쌍이 최대 4*N 개 → (B,N,N) 큰 텐서 없이
scatter 로 효율 계산한다. mcm_dst/mcm_src 가 그 쌍이다.
"""
import numpy as np
from collections import Counter

LEVELS = (1, 2, 3, 4)


def parse_ec(s):
    # multi-label 구분자: 콤마와 세미콜론(팀 전처리) 둘 다 처리
    return [e.strip() for e in str(s).replace(';', ',').split(',') if e.strip()]


def known_depth(ec):
    d = 0
    for p in ec.split('.'):
        if p == '-':
            break
        d += 1
    return d


def truncate(ec, L):
    return '.'.join(ec.split('.')[:L])


def build_hierarchy(ec_strings, min_freq=1):
    """train EC 문자열들 -> 노드/인덱스/레벨 + MCM scatter 쌍(dst=조상, src=자손)."""
    cnt = Counter()
    for s in ec_strings:
        for e in parse_ec(s):
            d = known_depth(e)
            for L in LEVELS:
                if d >= L:
                    cnt[truncate(e, L)] += 1
    nodes = sorted([k for k, c in cnt.items() if c >= min_freq],
                   key=lambda x: (x.count('.'), x))
    idx = {n: i for i, n in enumerate(nodes)}
    level = np.array([n.count('.') + 1 for n in nodes], dtype=np.int64)
    N = len(nodes)

    # (조상 i, 자손 j) 쌍: j 가 i 로 시작하는 부분 EC (자기 자신 포함)
    dst, src = [], []
    # 조상 빠른 조회: 각 노드의 접두 노드들
    for j, nj in enumerate(nodes):
        parts = nj.split('.')
        for L in range(1, len(parts) + 1):
            anc = '.'.join(parts[:L])
            if anc in idx:
                dst.append(idx[anc])   # 조상 i
                src.append(j)          # 자손 j
    return {'nodes': nodes, 'idx': idx, 'level': level, 'N': N,
            'mcm_dst': np.array(dst, dtype=np.int64),
            'mcm_src': np.array(src, dtype=np.int64)}


def encode_labels(ec_strings, hier):
    """샘플별 멀티핫 Y(조상 폐포) + 레벨 마스크 M(그 레벨을 아는지).

    불완전 EC('1.1.-.-')는 아는 레벨(1,2) 노드만 1, 모르는 레벨은 마스크로 손실에서 제외.
    """
    idx, N = hier['idx'], hier['N']
    Y = np.zeros((len(ec_strings), N), dtype=np.float32)
    M = np.zeros((len(ec_strings), len(LEVELS)), dtype=np.float32)
    for r, s in enumerate(ec_strings):
        for e in parse_ec(s):
            d = known_depth(e)
            for L in LEVELS:
                if d >= L:
                    M[r, L - 1] = 1.0
                    node = truncate(e, L)
                    if node in idx:
                        Y[r, idx[node]] = 1.0
    return Y, M
