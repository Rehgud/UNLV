#!/usr/bin/env python3
"""전처리 CSV(팀원 제공: Entry/Sequence/EC number) -> ECPICK 파이프라인 포맷(id/seq/ec_number).

ECPICK 파이프라인은 df['seq'], df['ec_number'] 컬럼을 그대로 읽는다(ec_ecpick_pipeline.py).
팀원 전처리본의 컬럼명만 표준화해서 data/ 에 저장한다. 서열/라벨 내용은 손대지 않는다.

사용:
    python prep_data.py                       # 기본 경로에서 train/val/test 변환
    python prep_data.py --src ../ECPICK/data  # 원본 CSV 위치 지정
"""
import argparse
import os
import pandas as pd

# 원본 컬럼 -> 표준 컬럼 (대소문자/공백 무시하고 매칭)
ALIASES = {
    'id':        ['id', 'entry', 'accession', 'uniprot'],
    'seq':       ['seq', 'sequence', 'protein_sequence'],
    'ec_number': ['ec_number', 'ec number', 'ec', 'ec_numbers'],
}


def _norm(c):
    return str(c).strip().lower()


def standardize(df):
    lut = {_norm(c): c for c in df.columns}
    out = {}
    for std, alist in ALIASES.items():
        hit = next((lut[a] for a in alist if a in lut), None)
        if hit is not None:
            out[std] = df[hit]
    if 'seq' not in out or 'ec_number' not in out:
        raise SystemExit(
            f'seq/ec_number 컬럼을 찾지 못함. 원본 컬럼={list(df.columns)}')
    res = pd.DataFrame(out)
    if 'id' not in res:
        res.insert(0, 'id', [f'p{i}' for i in range(len(res))])
    # 빈 서열/라벨 제거
    before = len(res)
    res = res.dropna(subset=['seq', 'ec_number'])
    res = res[(res['seq'].str.len() > 0) & (res['ec_number'].str.len() > 0)]
    dropped = before - len(res)
    return res[['id', 'seq', 'ec_number']].reset_index(drop=True), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(os.path.dirname(__file__), '..', 'ECPICK', 'data'),
                    help='원본 전처리 CSV 디렉터리 (train_ec/val_ec/test_ec.csv)')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'data'),
                    help='출력 디렉터리')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for split in ['train_ec', 'val_ec', 'test_ec']:
        src = os.path.join(args.src, f'{split}.csv')
        if not os.path.exists(src):
            print(f'  [skip] {src} 없음')
            continue
        df = pd.read_csv(src)
        std, dropped = standardize(df)
        dst = os.path.join(args.out, f'{split}.csv')
        std.to_csv(dst, index=False)
        n_ec = std['ec_number'].nunique()
        print(f'  {split}: {len(std):>7} rows (drop {dropped}) | uniq EC={n_ec} -> {dst}')

    print('완료.')


if __name__ == '__main__':
    main()
