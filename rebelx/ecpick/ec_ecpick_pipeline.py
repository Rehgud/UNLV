#!/usr/bin/env python3
"""
================================================================================
 ECPICK 재구현 파이프라인 (한 파일) — 우리 데이터(EC-Bench 등)로 학습/평가
================================================================================
 단백질 서열 -> one-hot(1000×21) -> ECPICK CNN(3 conv + 계층적 global/local) -> per-level F1.

 저자 공식 코드(datax-lab/ECPICK, model.py)의 아키텍처를 충실 재현.
 단, KAN 파이프라인과 **같은 데이터·라벨·지표·불완전EC 마스킹**을 써서 공정 비교되게 함.
 → ESM 불필요(서열에서 바로 학습). run_training/eval 인터페이스는 ec_kan_pipeline.py 와 동일.

 [ECPICK 구조 요약]
   서열 one-hot (B,1,1000,21)
     ├─ conv(k=4,21)  -> ReLU -> maxpool -> 128
     ├─ conv(k=8,21)  -> ReLU -> maxpool -> 128   ── concat -> 384-d 특징
     └─ conv(k=16,21) -> ReLU -> maxpool -> 128
   특징(384)
     ├─ Global flow : 레벨별 hidden 체인 -> 전체 클래스 출력(Sigmoid)
     └─ Local flow  : 레벨별 hidden+출력(Sigmoid), global/이전local 과 연결
   final = beta*global + (1-beta)*local     (beta=0.6)
   loss  = BCE(global) + BCE(local)          (여기선 불완전 EC 레벨 마스킹 추가)

 [사용 예]
   python ec_ecpick_pipeline.py                        # EC-Bench, 10-fold + 고정분할
   python ec_ecpick_pipeline.py --eval kfold --kfold 10
   python ec_ecpick_pipeline.py --data-dir data/우리데이터   # 전처리 데이터로 교체
   python ec_ecpick_pipeline.py --max-train 5000 --epochs 3  # 빠른 점검

 [설치] pip install torch scikit-learn pandas numpy    (ESM/efficient-kan 불필요)
================================================================================
"""
import argparse, os, time
import numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import torch, torch.nn as nn, torch.nn.functional as F
import torch.multiprocessing as _mp
from torch.utils.data import Dataset, DataLoader

# FD 누수 방지: 워커가 텐서를 '파일 디스크립터'로 공유하면 loader 를 수백 번 만드는
# 앙상블×반복 루프에서 열린 FD 가 계속 쌓여 [Errno 24] Too many open files 로 죽는다.
# 'file_system' 전략은 이름으로 공유해 FD 수가 일정하게 유지됨(표준 해결책).
try:
    _mp.set_sharing_strategy('file_system')
except (RuntimeError, AttributeError):
    pass

LEVELS = (1, 2, 3, 4)
AA = 'ACDEFGHIKLMNPQRSTVWYX'                 # 21종 (X=미지/기타). ECPICK seq_encoder 와 동일
AA_IDX = {c: i for i, c in enumerate(AA)}
MAX_LEN = 1000                              # ECPICK 입력 서열 길이(초과 자름, 미만 0패딩)
PAD = 99                                    # 패딩 위치 표식(one-hot 전부 0)


def pick_device(opt):
    if opt: return opt
    if torch.cuda.is_available(): return 'cuda'
    if torch.backends.mps.is_available(): return 'mps'
    return 'cpu'


# ----------------------------- 라벨 처리 (KAN 파이프라인과 동일) -----------------------------
def parse_ec(s):
    # multi-label 구분자: 콤마(EC-Bench)와 세미콜론(팀 전처리 train_ec) 둘 다 처리
    return [e.strip() for e in str(s).replace(';', ',').split(',') if e.strip()]

def known_depth(ec):
    d = 0
    for p in ec.split('.'):
        if p == '-': break
        d += 1
    return d

def truncate_ec(ec, L):
    return '.'.join(ec.split('.')[:L])

def build_vocab(train_df, min_freq):
    """레벨별 클래스 어휘(train 기준) + 클래스별 학습 개수(train_count)."""
    level_vocab, level_idx, level_count = {}, {}, {}
    for L in LEVELS:
        cnt = Counter()
        for s in train_df['ec_number']:
            for e in parse_ec(s):
                if known_depth(e) >= L:
                    cnt[truncate_ec(e, L)] += 1
        labs = sorted([k for k, c in cnt.items() if c >= min_freq])
        level_vocab[L] = labs
        level_idx[L] = {lab: i for i, lab in enumerate(labs)}
        level_count[L] = {lab: cnt[lab] for lab in labs}     # 클래스별 학습 예시 수
    return level_vocab, level_idx, level_count

def encode_labels(df, level_idx):
    """샘플별 {레벨: [클래스인덱스,...]}. '아는 레벨'만 키로(불완전 EC 마스킹용)."""
    out = []
    for s in df['ec_number']:
        ecs = parse_ec(s); per = {}
        for L in LEVELS:
            idxs, known = set(), False
            for e in ecs:
                if known_depth(e) >= L:
                    known = True
                    lab = truncate_ec(e, L)
                    if lab in level_idx[L]:
                        idxs.add(level_idx[L][lab])
            if known:
                per[L] = sorted(idxs)
        out.append(per)
    return out


# ----------------------------- 서열 one-hot 인코딩 -----------------------------
def encode_seqs(seqs):
    """서열 리스트 -> (N, MAX_LEN) uint8 인덱스 배열. 0~20=AA, PAD=패딩(one-hot 0).
    (실제 one-hot 은 배치에서 만들어 메모리 절약)."""
    N = len(seqs)
    idx = np.full((N, MAX_LEN), PAD, dtype=np.uint8)
    for n, s in enumerate(seqs):
        s = s[:MAX_LEN]
        for i, c in enumerate(s):
            idx[n, i] = AA_IDX.get(c, AA_IDX['X'])   # 모르는 문자는 X
    return idx

def batch_onehot(idx_batch):
    """(B, MAX_LEN) 인덱스 -> (B, 1, MAX_LEN, 21) one-hot float32 (벡터화 scatter)."""
    B = idx_batch.shape[0]
    oh = np.zeros((B, MAX_LEN, len(AA)), dtype=np.float32)
    valid = idx_batch < len(AA)                 # PAD(99)는 제외 → 전부 0
    bb, pp = np.where(valid)
    oh[bb, pp, idx_batch[bb, pp]] = 1.0
    return oh[:, None, :, :]                     # 채널축 추가 (B,1,MAX_LEN,21)


# ----------------------------- Dataset / collate -----------------------------
class ECPICKDataset(Dataset):
    def __init__(self, X_idx, labels):
        self.X_idx = X_idx; self.labels = labels
    def __len__(self):  return len(self.X_idx)
    def __getitem__(self, i):  return self.X_idx[i], self.labels[i]

def make_collate(level_sizes):
    """배치: one-hot 입력 + 전체 레벨 이어붙인 타깃 y + 레벨 마스크 lvlmask 생성."""
    starts = np.cumsum([0] + level_sizes)       # 레벨별 컬럼 시작 위치
    total = int(starts[-1])
    def collate(batch):
        idx = np.stack([b[0] for b in batch])
        x = torch.from_numpy(batch_onehot(idx)) # (B,1,MAX_LEN,21)
        B = len(batch)
        y = torch.zeros(B, total)               # 전체 클래스 멀티핫 (레벨 이어붙임)
        lvlmask = torch.zeros(B, len(LEVELS))   # 레벨별 '아는지' 마스크
        for b, (_, per) in enumerate(batch):
            for li, L in enumerate(LEVELS):
                if L in per:
                    lvlmask[b, li] = 1.0
                    for c in per[L]:
                        y[b, starts[li] + c] = 1.0
        return x, y, lvlmask
    return collate, starts, total


# ----------------------------- ECPICK 모델 (충실 재현) -----------------------------
class Flatten(nn.Module):
    def forward(self, x):  return x.view(x.size(0), -1)

class ECPICK(nn.Module):
    """3개 병렬 CNN + 계층적 global/local flow. 저자 model.py 를 그대로 옮김."""
    def __init__(self, level_sizes, relu_size=384, dropout=0.8, beta=0.6):
        super().__init__()
        self.level_sizes = level_sizes
        self.beta = beta
        mom, eps = 0.99, 1e-01
        def conv(k):                            # (B,1,1000,21) -> 128
            return nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=(k, len(AA)), stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(MAX_LEN - k + 1, 1)),
                Flatten(),
                nn.BatchNorm1d(128, momentum=mom, eps=eps))
        self.cnn1, self.cnn2, self.cnn3 = conv(4), conv(8), conv(16)
        feat = 384                              # 128*3

        # Global flow: 레벨별 hidden 체인
        self.global_hidden = nn.ModuleList([
            nn.Sequential(nn.Linear(feat if i == 0 else feat + relu_size, relu_size),
                          nn.ReLU(), nn.BatchNorm1d(relu_size, momentum=mom, eps=eps), nn.Dropout(dropout))
            for i in range(len(level_sizes))])
        self.global_out = nn.Sequential(nn.Linear(feat + relu_size, sum(level_sizes)), nn.Sigmoid())

        # Local flow: 레벨별 hidden + 출력
        self.local_hidden = nn.ModuleList()
        self.local_out = nn.ModuleList()
        for i in range(len(level_sizes)):
            in_size = feat + relu_size if i == 0 else feat + relu_size + level_sizes[i - 1]
            self.local_hidden.append(nn.Sequential(
                nn.Linear(in_size, relu_size), nn.ReLU(),
                nn.BatchNorm1d(relu_size, momentum=mom, eps=eps), nn.Dropout(dropout)))
            self.local_out.append(nn.Sequential(nn.Linear(relu_size, level_sizes[i]), nn.Sigmoid()))

    def forward(self, x):
        feat = torch.cat([self.cnn1(x), self.cnn2(x), self.cnn3(x)], dim=1)   # (B,384)
        # Global
        gh = []
        for i in range(len(self.global_hidden)):
            inp = feat if i == 0 else torch.cat([feat, gh[-1]], dim=1)
            gh.append(self.global_hidden[i](inp))
        global_out = self.global_out(torch.cat([feat, gh[-1]], dim=1))        # (B,total)
        # Local
        lo = []
        for i in range(len(self.local_hidden)):
            inp = torch.cat([feat, gh[i]], dim=1) if i == 0 else torch.cat([feat, gh[i], lo[-1]], dim=1)
            lo.append(self.local_out[i](self.local_hidden[i](inp)))
        local_out = torch.cat(lo, dim=1)                                     # (B,total)
        final_all = self.beta * global_out + (1 - self.beta) * local_out
        return global_out, local_out, final_all


def compute_pos_weight(lab_tr, level_sizes, cap=50.0):
    """클래스별 양성 가중치 pos_weight = neg/pos (레벨별 '아는 샘플' 기준), [1,cap] 클리핑.
    논문 §S4 'class weights(불균형 완화)'의 실제 의미. 이게 없으면 극단적 멀티라벨
    불균형이 모든 sigmoid 출력을 0.5 아래로 눌러 → threshold 0.5에서 F1=0 (all-negative)."""
    starts = np.cumsum([0] + list(level_sizes)); total = int(starts[-1])
    pos = np.zeros(total, dtype='float64'); known = np.zeros(len(LEVELS))
    for per in lab_tr:
        for li, L in enumerate(LEVELS):
            if L in per:
                known[li] += 1
                for c in per[L]:
                    pos[starts[li] + c] += 1
    pw = np.ones(total, dtype='float64')
    for li in range(len(LEVELS)):
        s, e = starts[li], starts[li + 1]
        neg = known[li] - pos[s:e]
        pw[s:e] = np.clip(neg / np.clip(pos[s:e], 1.0, None), 1.0, cap)   # 희소 양성일수록 큰 가중
    return pw.astype('float32')


def masked_bce(pred, y, colmask, posw=None):
    """마스킹된 BCE 합 평균. colmask=1인 (샘플,클래스)만 반영(불완전 EC 안전).
    posw(pos_weight, shape=total) 주면 양성 칸을 pos_weight배로 가중 → 불균형 상쇄,
    확률이 [0,1] 전체로 캘리브레이션(양성 확률이 0.5 위로 올라와 예측이 생김)."""
    loss = F.binary_cross_entropy(pred.clamp(1e-7, 1 - 1e-7), y, reduction='none')
    if posw is not None:
        loss = loss * (y * posw + (1.0 - y))   # 양성칸=pos_weight, 음성칸=1 (브로드캐스트)
    return (loss * colmask).sum() / colmask.sum().clamp(min=1.0)


@torch.no_grad()
def predict_probs(model, loader, device, starts, col2level=None, posw=None):
    """eval셋 예측 확률(final_all=sigmoid) + 정답 + 레벨마스크 (레벨별 concat, 임계값 전).
    앙상블: 여러 모델의 이 확률을 평균한 뒤 임계값 적용.
    col2level(+posw) 주면 같은 한 번의 순회에서 val loss(train과 동일한 masked BCE)도 계산해 4번째로 반환."""
    model.eval()
    P = {L: [] for L in LEVELS}; Y = {L: [] for L in LEVELS}; M = {L: [] for L in LEVELS}
    loss_sum = 0.0; n_seen = 0
    for x, y, lvlmask in loader:
        g, l, final_all = model(x.to(device))
        if col2level is not None:                                  # train 과 동일한 손실로 val loss 계산
            yd = y.to(device); colmask = lvlmask[:, col2level].to(device)
            loss_sum += (masked_bce(g, yd, colmask, posw) + masked_bce(l, yd, colmask, posw)).item() * len(x)
            n_seen += len(x)
        pr = final_all.cpu().numpy(); yn = y.numpy().astype(int); mm = lvlmask.numpy().astype(bool)
        for li, L in enumerate(LEVELS):
            s, e = starts[li], starts[li + 1]
            P[L].append(pr[:, s:e]); Y[L].append(yn[:, s:e]); M[L].append(mm[:, li])
    val_loss = (loss_sum / n_seen) if (col2level is not None and n_seen) else None
    return ({L: np.vstack(P[L]) for L in LEVELS},
            {L: np.vstack(Y[L]) for L in LEVELS},
            {L: np.concatenate(M[L]) for L in LEVELS},
            val_loss)


def _f1str(df):
    """F1 데이터프레임 -> 'L1=macro/micro L2=... ' 한 줄(실시간 출력용). macro/micro 둘 다 표시."""
    return ' '.join(f"L{int(r.level)}={r.macroF1:.3f}/{r.microF1:.3f}" for r in df.itertuples())


METRIC_COLS = ['macroP', 'macroR', 'macroF1', 'microP', 'microR', 'microF1']


def f1_from_probs(P, Y, M, thr):
    """레벨별 (평균)확률 -> 임계값 -> per-level Precision/Recall/F1 (macro & micro).
    교수님·논문과 동일하게 P/R/F1 모두 보고. '그 레벨 아는 샘플'만 대상."""
    rows = []
    for L in LEVELS:
        m = M[L]
        if m.sum() == 0: continue
        p = (P[L][m] > thr).astype(int); y = Y[L][m]
        rows.append({'level': L, 'n_eval': int(m.sum()),
                     'n_labeled': int((y.sum(axis=1) > 0).sum()),
                     'macroP': precision_score(y, p, average='macro', zero_division=0),
                     'macroR': recall_score(y, p, average='macro', zero_division=0),
                     'macroF1': f1_score(y, p, average='macro', zero_division=0),
                     'microP': precision_score(y, p, average='micro', zero_division=0),
                     'microR': recall_score(y, p, average='micro', zero_division=0),
                     'microF1': f1_score(y, p, average='micro', zero_division=0)})
    return pd.DataFrame(rows)


def per_class_metrics(P, Y, M, level_vocab, level_count, thr):
    """클래스별 P/R/F1 + 학습개수(train_count) + test 지원수. 저빈도 클래스 분석용.
    P/Y/M: 레벨별 (앙상블 평균)확률/정답/레벨마스크. '그 레벨 아는 샘플'만 대상."""
    rows = []
    for L in LEVELS:
        m = M[L]
        if m.sum() == 0:
            continue
        Pm = (P[L][m] > thr).astype(int)
        Ym = Y[L][m].astype(int)
        vocab = level_vocab[L]
        for j, ec in enumerate(vocab):
            pred = Pm[:, j]; true = Ym[:, j]
            tp = int((pred & true).sum()); fp = int((pred & (1 - true)).sum()); fn = int(((1 - pred) & true).sum())
            support = tp + fn                                  # 이 클래스의 test 정답 수
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            rows.append({'level': L, 'ec_class': ec,
                         'train_count': int(level_count[L].get(ec, 0)),
                         'test_support': support, 'TP': tp, 'FP': fp, 'FN': fn,
                         'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4)})
    return pd.DataFrame(rows)


def train_one(X_tr, lab_tr, level_sizes, args, device, seed, verbose, posw=None, el=None, starts=None):
    """앙상블 멤버 1개 학습(멤버마다 랜덤 초기화 다름). 논문: Adam eps=1e-7.
    posw=pos_weight 텐서(있으면 불균형 상쇄). el/starts 주면 val_every epoch마다 val F1 출력."""
    collate, _, _ = make_collate(level_sizes)
    tl = DataLoader(ECPICKDataset(X_tr, lab_tr), batch_size=args.batch, shuffle=True,
                    collate_fn=collate, drop_last=(len(X_tr) > args.batch),
                    num_workers=args.num_workers, pin_memory=(str(device) == 'cuda'),
                    persistent_workers=(args.num_workers > 0))
    torch.manual_seed(seed)                                        # 멤버마다 다른 초기화 = 앙상블 다양성
    model = ECPICK(level_sizes, args.relu_size, args.dropout, args.beta).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, eps=1e-7)
    col2level = torch.from_numpy(np.concatenate([[li] * n for li, n in enumerate(level_sizes)])).long()
    t0 = time.time()
    hist = []                                                     # 러닝커브(epoch별 train/val loss + 레벨별 macro/micro F1)
    for ep in range(1, args.epochs + 1):
        model.train(); run = 0.0
        for x, y, lvlmask in tl:
            x, y = x.to(device), y.to(device)
            colmask = lvlmask[:, col2level].to(device)
            g, l, _ = model(x)
            loss = masked_bce(g, y, colmask, posw) + masked_bce(l, y, colmask, posw)
            optim.zero_grad(); loss.backward(); optim.step()
            run += loss.item() * len(x)
        tr_loss = run / len(tl.dataset)
        rec = {'epoch': ep, 'train_loss': tr_loss, 'val_loss': np.nan}
        do_val = el is not None and args.val_every and (ep % args.val_every == 0 or ep == args.epochs)
        if do_val:                                               # epoch 중 val loss + 레벨별 macro/micro F1
            P, Y, M, vloss = predict_probs(model, el, device, starts, col2level, posw)
            fdf = f1_from_probs(P, Y, M, args.threshold)
            rec['val_loss'] = vloss
            for r in fdf.itertuples():
                rec[f'L{int(r.level)}_macroF1'] = r.macroF1
                rec[f'L{int(r.level)}_microF1'] = r.microF1
            model.train()
        hist.append(rec)
        if verbose:
            msg = f'    epoch {ep:2d}/{args.epochs}  train_loss={tr_loss:.4f}'
            if do_val:
                msg += f'  val_loss={vloss:.4f}'
            msg += f'  ({(time.time()-t0)/60:.1f}분)'
            if do_val:
                msg += '  | val F1(macro/micro) ' + _f1str(fdf)
            print(msg, flush=True)
    # persistent_workers 워커/FD 를 멤버마다 확실히 회수(루프 누적 방지)
    if getattr(tl, '_iterator', None) is not None:
        tl._iterator._shutdown_workers()
    del tl
    return model, pd.DataFrame(hist)


def fit_and_eval(X_tr, lab_tr, X_te, lab_te, level_sizes, args, device, verbose=True, curve_tag=None,
                 per_class_out=None, level_vocab=None, level_count=None, probs_out=None):
    """ECPICK 앙상블: N개 모델(랜덤초기화) 학습 -> eval 확률 평균 -> per-level F1 (논문 N=10).
    curve_tag 주면 멤버별 epoch 러닝커브(train/val loss + macro/micro F1)를 CSV 로 저장."""
    collate, starts, _ = make_collate(level_sizes)
    el = DataLoader(ECPICKDataset(X_te, lab_te), batch_size=args.batch, shuffle=False, collate_fn=collate,
                    num_workers=args.num_workers, pin_memory=(str(device) == 'cuda'),
                    persistent_workers=(args.num_workers > 0))
    posw = None
    if args.class_weights:                                         # 논문 §S4: pos_weight로 불균형 상쇄(all-zero 방지)
        posw = torch.from_numpy(compute_pos_weight(lab_tr, level_sizes, args.pw_cap)).to(device)
    n_ens = max(1, args.ensemble)
    sumP = None; Y = M = None; curves = []
    for mi in range(n_ens):
        if verbose: print(f'  [ensemble {mi + 1}/{n_ens}] 학습...')
        model, hist = train_one(X_tr, lab_tr, level_sizes, args, device,
                                seed=args.seed * 1000 + mi, verbose=verbose, posw=posw, el=el, starts=starts)
        hist.insert(0, 'ensemble', mi + 1); curves.append(hist)   # 멤버별 러닝커브 누적
        P, Y, M, _ = predict_probs(model, el, device, starts)     # Y,M 은 멤버마다 동일(같은 eval셋)
        sumP = P if sumP is None else {L: sumP[L] + P[L] for L in LEVELS}
        if verbose:                                               # 멤버별 + 누적 앙상블 val F1(macro/micro) 실시간 출력
            solo = f1_from_probs(P, Y, M, args.threshold)
            ens = f1_from_probs({L: sumP[L] / (mi + 1) for L in LEVELS}, Y, M, args.threshold)
            print('    val F1(macro/micro)  model[' + _f1str(solo) + ']  ensemble[' + _f1str(ens) + ']')
        del model
        if str(device) == 'cuda': torch.cuda.empty_cache()
    avgP = {L: sumP[L] / n_ens for L in LEVELS}                    # posterior 평균 = 앙상블 예측
    if probs_out is not None:                                      # 테스트 확률 저장 → threshold sweep 로컬 분석용
        os.makedirs(os.path.dirname(probs_out) or '.', exist_ok=True)
        np.savez_compressed(probs_out,
                            **{f'P{L}': avgP[L].astype('float32') for L in LEVELS},
                            **{f'Y{L}': Y[L].astype('int8') for L in LEVELS},
                            **{f'M{L}': M[L].astype(bool) for L in LEVELS})
        if verbose: print(f'  [테스트 확률 저장] {probs_out}', flush=True)
    result = f1_from_probs(avgP, Y, M, args.threshold)
    # 클래스별 성능 + 학습개수 저장(저빈도 클래스 분석용)
    if per_class_out is not None and level_vocab is not None and level_count is not None:
        pc = per_class_metrics(avgP, Y, M, level_vocab, level_count, args.threshold)
        os.makedirs(os.path.dirname(per_class_out) or '.', exist_ok=True)
        pc.to_csv(per_class_out, index=False)
        if verbose: print(f'  [클래스별 성능 저장] {per_class_out} ({len(pc)} classes)')
    # 러닝커브 저장(교수님 보고용) — 멤버별 epoch history + 멤버평균
    if curve_tag is not None and curves:
        os.makedirs(args.results_dir, exist_ok=True)
        cdf = pd.concat(curves, ignore_index=True)
        cpath = f'{args.results_dir}/learncurve__{curve_tag}.csv'
        cdf.to_csv(cpath, index=False)
        mean_cols = [c for c in cdf.columns if c not in ('ensemble',)]
        cdf.groupby('epoch')[[c for c in mean_cols if c != 'epoch']].mean().reset_index().to_csv(
            f'{args.results_dir}/learncurve__{curve_tag}__mean.csv', index=False)
        if verbose: print(f'  [러닝커브 저장] {cpath} (+ __mean.csv)')
    # eval 로더의 persistent 워커/FD 를 repeat 마다 회수(train 로더처럼) → [Errno 24] 누적 방지
    if getattr(el, '_iterator', None) is not None:
        el._iterator._shutdown_workers()
    del el
    return result


def run_training(args, device):
    # ---- 데이터 ----
    data_dir = args.data_dir or next(
        (d for d in ['data/EC-Bench_data', 'EC-Bench_data', '../data/EC-Bench_data'] if os.path.isdir(d)), 'data/EC-Bench_data')
    train_csv = args.train_csv or f'{data_dir}/train_100.csv'
    test_csv = args.test_csv or f'{data_dir}/test_ec.csv'
    print(f'[data] {train_csv}' + ('' if args.eval == 'kfold' else f' | {test_csv}'))

    train_df = pd.read_csv(train_csv)
    if args.max_train:
        train_df = train_df.sample(n=min(args.max_train, len(train_df)), random_state=args.seed).reset_index(drop=True)

    # ---- 라벨 + 서열 인코딩 ----
    level_vocab, level_idx, level_count = build_vocab(train_df, args.min_freq)
    level_sizes = [len(level_vocab[L]) for L in LEVELS]
    print('레벨별 클래스:', level_sizes, '| 총출력:', sum(level_sizes))
    train_lab = encode_labels(train_df, level_idx)
    print('서열 one-hot 인덱스 인코딩...'); X_train = encode_seqs(train_df['seq'].tolist())

    os.makedirs(args.results_dir, exist_ok=True)
    train_name = os.path.basename(train_csv)
    base = f'ECPICK__seq__{train_name.replace(".csv", "")}'

    # ---- (0) repeated stratified hold-out (교수님·HIT-EC 방식) ----
    # train 을 매번 층화(레벨1 EC 비율 유지) 80/20 로 나눠 R번 반복 → 레벨별 F1 평균±표준편차. HIT-EC는 R=10.
    if args.eval == 'rsho':
        from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
        strat = np.array([per[1][0] if (1 in per and per[1]) else -1 for per in train_lab])  # 레벨1 EC로 층화
        try:
            sp = StratifiedShuffleSplit(n_splits=args.repeats, test_size=args.test_size, random_state=args.seed)
            splits = list(sp.split(np.zeros(len(X_train)), strat))
        except ValueError as e:
            print(f'  (층화 불가 → 무작위 hold-out 대체: {e})')
            splits = list(ShuffleSplit(n_splits=args.repeats, test_size=args.test_size,
                                       random_state=args.seed).split(np.zeros(len(X_train))))
        perrep_path = f'{args.results_dir}/{base}__rsho{args.repeats}__perrep.csv'
        reps = []; done_repeats = set()
        if os.path.exists(perrep_path):                          # resume: 이미 끝난 repeat 이어받기(죽어도 안전)
            prev = pd.read_csv(perrep_path); reps.append(prev)
            done_repeats = set(int(r) for r in prev['repeat'].unique())
            print(f'  [resume] 이미 완료된 repeat: {sorted(done_repeats)} → 나머지만 진행')
        for ri, (tr, te) in enumerate(splits, 1):
            if ri in done_repeats:
                continue                                         # 이미 한 반복은 건너뜀
            print(f'\n--- repeat {ri}/{args.repeats} (train {len(tr)} / test {len(te)}) ---')
            res = fit_and_eval(X_train[tr], [train_lab[i] for i in tr],
                               X_train[te], [train_lab[i] for i in te],
                               level_sizes, args, device, verbose=True,
                               curve_tag=f'{base}__rsho{args.repeats}__rep{ri}')   # epoch 실시간 출력 + 러닝커브 저장
            res.insert(0, 'repeat', ri); reps.append(res)
            pd.concat(reps, ignore_index=True).to_csv(perrep_path, index=False)   # 반복마다 저장
            print(res[['level', 'n_eval', 'macroP', 'macroR', 'macroF1', 'microP', 'microR', 'microF1']].round(4).to_string(index=False))
        allr = pd.concat(reps, ignore_index=True)
        agg = {f'{mc}_{st}': (mc, st) for mc in METRIC_COLS for st in ('mean', 'std')}
        summ = allr.groupby('level').agg(**agg).reset_index()
        tp, ep = int((1 - args.test_size) * 100), int(args.test_size * 100)
        print(f'\n=== {args.repeats}× repeated stratified hold-out (train{tp}/test{ep}) — 평균±표준편차 ===')
        print(summ.round(4).to_string(index=False))
        allr.to_csv(f'{args.results_dir}/{base}__rsho{args.repeats}__perrep.csv', index=False)
        summ.to_csv(f'{args.results_dir}/{base}__rsho{args.repeats}.csv', index=False)
        print(f'saved -> {args.results_dir}/{base}__rsho{args.repeats}.csv (+ __perrep.csv)')
        return

    do_kfold = args.eval in ('kfold', 'both') and args.kfold > 1
    do_fixed = args.eval in ('fixed', 'both')

    # ---- (1) k-fold CV ----
    if do_kfold:
        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        folds = []
        for fi, (tr, va) in enumerate(kf.split(np.arange(len(X_train))), 1):
            print(f'\n--- fold {fi}/{args.kfold} (train {len(tr)} / val {len(va)}) ---')
            res = fit_and_eval(X_train[tr], [train_lab[i] for i in tr],
                               X_train[va], [train_lab[i] for i in va],
                               level_sizes, args, device, verbose=True,
                               curve_tag=f'{base}__cv{args.kfold}__fold{fi}')
            res.insert(0, 'fold', fi); folds.append(res)
            print(res[['level', 'n_eval', 'macroP', 'macroR', 'macroF1', 'microP', 'microR', 'microF1']].round(4).to_string(index=False))
        allf = pd.concat(folds, ignore_index=True)
        agg = {f'{mc}_{st}': (mc, st) for mc in METRIC_COLS for st in ('mean', 'std')}
        summary = allf.groupby('level').agg(**agg).reset_index()
        print(f'\n=== {args.kfold}-fold CV — 레벨별 평균±표준편차 ===')
        print(summary.round(4).to_string(index=False))
        allf.to_csv(f'{args.results_dir}/{base}__cv{args.kfold}__perfold.csv', index=False)
        summary.to_csv(f'{args.results_dir}/{base}__cv{args.kfold}.csv', index=False)
        print(f'saved -> {args.results_dir}/{base}__cv{args.kfold}.csv (+ __perfold.csv)')

    # ---- (2) 고정 분할 ----
    if do_fixed:
        test_df = pd.read_csv(test_csv)
        test_lab = encode_labels(test_df, level_idx)
        X_test = encode_seqs(test_df['seq'].tolist())
        res = fit_and_eval(X_train, train_lab, X_test, test_lab, level_sizes, args, device, verbose=True,
                           curve_tag=f'{base}__fixed',
                           per_class_out=f'{args.results_dir}/perclass__{base}__fixed.csv',
                           level_vocab=level_vocab, level_count=level_count,
                           probs_out=f'{args.results_dir}/testprobs__{base}__fixed.npz')
        print('\n=== test per-level F1 (고정 분할) ===')
        print(res.to_string(index=False))
        res.assign(model='ECPICK', train=train_name).to_csv(f'{args.results_dir}/{base}.csv', index=False)
        print(f'saved -> {args.results_dir}/{base}.csv')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default=None, help='data/EC-Bench_data 또는 EC-Bench_data 자동탐지. 전처리 데이터로 교체 가능')
    ap.add_argument('--train-csv', default=None)
    ap.add_argument('--test-csv', default=None)
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=0)
    # 평가
    ap.add_argument('--eval', choices=['both', 'kfold', 'fixed', 'rsho'], default='both',
                    help='both=kfold+fixed / kfold / fixed / rsho=반복 층화 hold-out(교수님·HIT-EC식)')
    ap.add_argument('--kfold', type=int, default=10)
    ap.add_argument('--repeats', type=int, default=10, help='rsho 반복 횟수 (HIT-EC=10)')
    ap.add_argument('--test-size', type=float, default=0.2, help='rsho 각 반복 test 비율(층화 80/20)')
    ap.add_argument('--min-freq', type=int, default=1, help='클래스 최소 등장수(KAN과 동일값으로 맞출 것)')
    ap.add_argument('--max-train', type=int, default=None, help='빠른 스모크용')
    # ECPICK 하이퍼파라미터 (저자 기본값)
    ap.add_argument('--relu-size', type=int, default=384)
    ap.add_argument('--dropout', type=float, default=0.8)
    ap.add_argument('--beta', type=float, default=0.6, help='final = beta*global + (1-beta)*local')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--num-workers', type=int, default=8, help='데이터로더 병렬 워커 (one-hot 생성 CPU 병목 완화)')
    ap.add_argument('--ensemble', type=int, default=10, help='앙상블 모델 수 (논문 ECPICK=10, 랜덤초기화 후 posterior 평균). 1=단일(빠름)')
    ap.add_argument('--class-weights', type=int, default=1, help='pos_weight로 불균형 상쇄(논문 §S4). 1=켬(기본) / 0=끔. 끄면 threshold 0.5에서 F1=0 위험')
    ap.add_argument('--pw-cap', type=float, default=50.0, help='pos_weight 상한(neg/pos 클리핑). 희소 클래스 과도가중 방지')
    ap.add_argument('--val-every', type=int, default=5, help='학습 중 N epoch마다 val F1 출력(0=끔)')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f'[device] {device} | ECPICK (seq CNN) | eval {args.eval}')
    run_training(args, device)
    print('\n✅ 끝.')


if __name__ == '__main__':
    main()
