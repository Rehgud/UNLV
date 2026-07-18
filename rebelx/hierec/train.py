#!/usr/bin/env python3
"""C-HMCNN (계층 인식 head) on ESM-3B features — EC 번호 예측.

MCM 으로 '자식 확률 <= 부모 확률' 을 강제(계층 일관) + MCLoss 로 학습.
팀원 flat 모델(MLP/CNN/Transformer)·CLEAN 과 같은 features/split/지표로 비교.

데이터 없이 지금 테스트:
    python train.py --synthetic --epochs 100 --eval-every 20
실제 데이터(포맷 맞춘 뒤):
    python train.py --features .../features_all.pt --split .../split.json --labels .../train.csv
"""
import argparse
import os
import time
import numpy as np
import torch

from hierarchy import LEVELS
from data import load_synthetic, load_real, make_split_arrays
from model import MLPBase, get_constr_out
from losses import mc_loss
from metrics import per_level_f1
from frequency import per_class_and_freq


def node_mask_from_lvl(M, level):
    """(B,4) 레벨 마스크 -> (B,N) 노드 마스크 (아는 레벨의 노드만 1)."""
    lvl = torch.as_tensor(level, device=M.device).long() - 1     # 0..3
    return M[:, lvl]


@torch.no_grad()
def predict_test(model, Xte_t, dst, src, N):
    """test MCM 확률 P(n,N) 반환 (per-level + 빈도분석 공용)."""
    model.eval()
    P = []
    for s in range(0, len(Xte_t), 1024):
        h = model(Xte_t[s:s + 1024])
        P.append(get_constr_out(h, dst, src, N).cpu().numpy())
    return np.vstack(P)


@torch.no_grad()
def evaluate(model, Xte_t, Yte, M_te, dst, src, N, level, thr):
    return per_level_f1(predict_test(model, Xte_t, dst, src, N), Yte, M_te, level, thr)


def run(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.synthetic:
        ids, X, ec, split = load_synthetic(args.n_synth, args.feat_dim, args.seed)
    else:
        ids, X, ec, split = load_real(args.features, args.split, args.labels, args.split_key)
    hier, (Xtr, Ytr, Mtr), (Xte, Yte, Mte) = make_split_arrays(ids, X, ec, split, args.min_freq)

    N, level = hier['N'], hier['level']
    dst = torch.tensor(hier['mcm_dst'], device=dev)
    src = torch.tensor(hier['mcm_src'], device=dev)
    print(f'[device] {dev} | nodes={N} '
          f'| level sizes={[int((level == L).sum()) for L in LEVELS]} '
          f'| train={len(Xtr)} test={len(Xte)}', flush=True)

    Xtr_t = torch.tensor(Xtr, device=dev)
    Ytr_t = torch.tensor(Ytr, device=dev)
    Mtr_t = torch.tensor(Mtr, device=dev)
    Xte_t = torch.tensor(Xte, device=dev)
    nmask = node_mask_from_lvl(Mtr_t, level) if args.mask_unknown else None

    # 노드별 pos_weight(불균형 상쇄) — 없으면 희귀 하위 노드가 threshold 0.5 못 넘어 L4=0 붕괴
    posw = None
    if args.class_weights:
        pos = Ytr.sum(0); neg = len(Ytr) - pos
        posw_np = np.clip(neg / np.clip(pos, 1.0, None), 1.0, args.pw_cap).astype('float32')
        posw = torch.tensor(posw_np, device=dev)
        print(f'[pos_weight] on (cap={args.pw_cap}) | 평균={posw_np.mean():.1f} 최대={posw_np.max():.1f}', flush=True)

    model = MLPBase(X.shape[1], N, args.hidden, args.layers, args.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    n = len(Xtr_t); t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(n, device=dev); tot = 0.0
        for s in range(0, n, args.batch):
            b = perm[s:s + args.batch]
            loss = mc_loss(model(Xtr_t[b]), Ytr_t[b], dst, src, N,
                           None if nmask is None else nmask[b], posw=posw)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if ep % args.eval_every == 0 or ep == args.epochs:
            df = evaluate(model, Xte_t, Yte, Mte, dst, src, N, level, args.threshold)
            s = ' '.join(f'L{int(r.level)}={r.microF1:.3f}' for r in df.itertuples())
            print(f'ep {ep:4d}/{args.epochs} loss={tot / n:.4f} '
                  f'({(time.time() - t0) / 60:.1f}m) | test microF1 {s}', flush=True)

    P = predict_test(model, Xte_t, dst, src, N)
    df = per_level_f1(P, Yte, Mte, level, args.threshold)
    print('\n=== per-level F1 (test) ===')
    print(df.round(4).to_string(index=False))
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or ('synth' if args.synthetic else 'real')
    path = f'{args.out}/hier_chmcnn__{tag}.csv'
    df.to_csv(path, index=False)
    print(f'saved -> {path}')

    # ── 클래스별 성능 + 학습개수 구간(빈도) 요약 — ECPICK 과 동일 스키마 ──
    train_count = np.asarray(Ytr).sum(0)                  # 노드별 train 양성 수 = 클래스별 학습개수
    perclass, freq = per_class_and_freq(P, Yte, Mte, level, hier['nodes'], train_count, args.threshold)
    pc_path = f'{args.out}/perclass__hier_chmcnn__{tag}.csv'
    fq_path = f'{args.out}/freq_summary__hier_chmcnn__{tag}.csv'
    perclass.to_csv(pc_path, index=False)
    freq.to_csv(fq_path, index=False)
    print(f'saved -> {pc_path} ({len(perclass)} classes)')
    print(f'saved -> {fq_path}')
    print('\n=== 학습개수 구간별 (test 등장 클래스) ===')
    print(freq.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--synthetic', action='store_true', help='데이터 없이 랜덤으로 파이프라인 테스트')
    ap.add_argument('--n-synth', type=int, default=4000)
    ap.add_argument('--feat-dim', type=int, default=2560, help='ESM-3B feature 차원')
    # 실제 데이터 (data.load_real 을 팀원 포맷에 맞춰 수정)
    ap.add_argument('--features', default=None)
    ap.add_argument('--split', default=None)
    ap.add_argument('--labels', default=None)
    ap.add_argument('--split-key', default=None, help='split.json 이 여러 similarity 버킷이면 키(예 70/30)')
    # 학습
    ap.add_argument('--hidden', type=int, default=1024)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--min-freq', type=int, default=1)
    ap.add_argument('--mask-unknown', type=int, default=1, help='불완전 EC 의 미지 레벨 노드를 손실에서 제외(ECPICK 방식)')
    ap.add_argument('--class-weights', type=int, default=1, help='pos_weight로 불균형 상쇄(1=켬). 끄면 하위 레벨 F1=0 위험')
    ap.add_argument('--pw-cap', type=float, default=50.0, help='pos_weight 상한(neg/pos 클리핑)')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--eval-every', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results')
    ap.add_argument('--tag', default=None, help='결과 파일 이름 태그(예 split70)')
    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    run(args)


if __name__ == '__main__':
    main()
