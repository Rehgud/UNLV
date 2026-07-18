"""데이터 로딩.

팀원의 features_all.pt + similarity split.json 은 아직 없어서, 실제 로더(load_real)는
인터페이스만 잡아두고 ★TODO★ 로 표시했다. --synthetic 로 랜덤 데이터를 만들어
model/loss/metric 파이프라인 전체를 지금 바로 검증할 수 있다.
데이터 오면 load_real 의 3개 매핑(feature/split/label)만 실제 포맷에 맞추면 된다.
"""
import json
import numpy as np
import torch
import pandas as pd
from hierarchy import build_hierarchy, encode_labels


def load_synthetic(n=4000, dim=2560, seed=0):
    """랜덤 ESM-3B 유사 feature + 랜덤 EC 라벨(파이프라인 smoke test 용)."""
    rng = np.random.default_rng(seed)
    ec_strings = []
    for _ in range(n):
        a, b, c, d = rng.integers(1, 7), rng.integers(1, 20), rng.integers(1, 15), rng.integers(1, 40)
        depth = rng.choice([2, 3, 4], p=[0.2, 0.3, 0.5])
        parts = [str(a), str(b), str(c), str(d)][:depth] + ['-'] * (4 - depth)
        ec_strings.append('.'.join(parts))
    X = rng.standard_normal((n, dim)).astype('float32')
    for i, s in enumerate(ec_strings):                     # feature 에 라벨 신호 살짝 심어 학습되게
        X[i, hash(s.split('.')[0]) % dim] += 3.0
    ids = [f's{i}' for i in range(n)]
    split = {'train': ids[:int(n * 0.8)], 'test': ids[int(n * 0.8):]}
    return ids, X, ec_strings, split


def load_real(feature_path, split_path, label_csv=None, split_key=None):
    """팀 ESM 추출본(features_all.pt) + split.json 로딩.

    반환: ids(list), X(np.ndarray[N,D]), ec_strings(list[str]), split({'train','test'})

    지원 포맷:
      A) 팀 형식(우선): dict{ids:[N], features:[N,D], targets:[N,C](멀티핫), labels:[C] EC문자열}
         → EC 문자열을 targets+labels 로 복원(라벨 CSV 불필요).
      B) 하위호환: dict{id:Tensor}(+label_csv) / Tensor[N,D](+label_csv)
      split.json: {'train':[...], ('valid':[...],) 'test':[...]} — id 목록. valid 는 사용 안 함(train→test).
    """
    blob = torch.load(feature_path, map_location='cpu')

    # ---------- A) 팀 형식: ids/features/targets/labels 한 파일에 ----------
    if isinstance(blob, dict) and 'features' in blob and 'ids' in blob:
        ids = [str(i) for i in blob['ids']]
        feats = blob['features']
        X = (feats.float().numpy() if hasattr(feats, 'numpy') else np.asarray(feats, dtype='float32'))
        if X.ndim == 3:                                    # per-residue 면 mean-pool
            X = X.mean(axis=1)
        if 'targets' in blob and 'labels' in blob:         # 멀티핫 -> EC 문자열 복원
            T = blob['targets'].numpy() if hasattr(blob['targets'], 'numpy') else np.asarray(blob['targets'])
            T = T.astype(bool)
            labs = np.asarray([str(l) for l in blob['labels']])
            ec_strings = [','.join(labs[row].tolist()) for row in T]
        elif label_csv:
            df = pd.read_csv(label_csv)
            idcol = 'id' if 'id' in df.columns else df.columns[0]
            eccol = 'ec_number' if 'ec_number' in df.columns else df.columns[-1]
            id2ec = dict(zip(df[idcol].astype(str), df[eccol].astype(str)))
            ec_strings = [id2ec.get(i, '') for i in ids]
        else:
            raise ValueError('targets/labels 도 없고 label_csv 도 없음 — 라벨 소스를 지정하세요.')

    # ---------- B) 하위호환: dict{id:Tensor} ----------
    elif isinstance(blob, dict):
        ids = [str(k) for k in blob.keys()]
        X = torch.stack([(v.mean(0) if v.dim() == 2 else v).float() for v in blob.values()]).numpy()
        ec_strings = _labels_from_csv(label_csv, ids)

    # ---------- B) 하위호환: Tensor[N,D] (+ split 의 id 순서 가정) ----------
    else:
        raise NotImplementedError('features 가 순수 Tensor 입니다. id 순서 파일이 필요합니다 — 팀 형식(dict) 사용 권장.')

    # ---------- split ----------
    split = json.load(open(split_path))
    if split_key is not None:
        split = split[split_key]
    split = {'train': [str(i) for i in split['train']],
             'test': [str(i) for i in split['test']]}      # valid 는 사용 안 함(ECPICK 처럼 train→test)
    return ids, X, ec_strings, split


def _labels_from_csv(label_csv, ids):
    df = pd.read_csv(label_csv)
    idcol = 'id' if 'id' in df.columns else df.columns[0]
    eccol = 'ec_number' if 'ec_number' in df.columns else df.columns[-1]
    id2ec = dict(zip(df[idcol].astype(str), df[eccol].astype(str)))
    return [id2ec.get(str(i), '') for i in ids]


def make_split_arrays(ids, X, ec_strings, split, min_freq=1):
    """train/test 로 나누고 라벨 인코딩. 계층은 train 라벨로만 구성."""
    id2i = {d: i for i, d in enumerate(ids)}
    tr = [id2i[d] for d in split['train'] if d in id2i]
    te = [id2i[d] for d in split['test'] if d in id2i]
    hier = build_hierarchy([ec_strings[i] for i in tr], min_freq)
    Y, M = encode_labels(ec_strings, hier)
    return hier, (X[tr], Y[tr], M[tr]), (X[te], Y[te], M[te])
