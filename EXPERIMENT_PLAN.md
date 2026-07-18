# Project3 실험 계획 (Experiment Plan)

> Deep Neural Networks for Protein Function Prediction
> 발표 자료(Method Overview / Weekly Plan)를 실제 실행 단계로 옮긴 문서. 팀 공유용.

## 0. 한 줄 요약

**frozen ESM 임베딩 → 여러 downstream head(MLP / CNN / Transformer / KAN) + 계층(hierarchical)
학습 → EC-Bench에서 per-level F1로 비교.**

- **목표 = 새 논문.** CLEAN / DeepEC(T) / ECPICK / HIT-EC / KAN(npj AI 2026)은 **베이스라인·토대**일 뿐,
  그대로 재현하는 게 아니라 *공유 ESM 임베딩 위 통제된 head 비교 + 우리 기여*를 만든다.
- **데이터 교체 가능**: 지금은 **EC-Bench 전처리본** 사용(공정 비교 + 품질 리스크 회피),
  나중에 **자체 전처리 UniProt**로 경로만 교체(스키마 `id,seq,ec_number` 동일).
- **임베딩 스케일업**: **650M 로컬 테스트 → 3B on UNLV REBELX → 이후 ESM3.**
  head 코드는 **입력 차원 자동 감지**라 모델 교체 시 변경 불필요.

---
 
## 1. 현재 위치 (Weekly Plan 기준)

| 단계 | 상태 |
|------|------|
| Project intro / SOTA survey / Data gathering | ✅ 완료 |
| Data preprocessing | ✅ **EC-Bench 전처리본 사용으로 해결** (자체 UniProt은 후순위 스왑) |
| **ESM 임베딩 추출** | 👉 진행 — 650M 로컬 → **3B는 REBELX** |
| Model Implementation / Training | 🟡 **KAN head 노트북 v1 완성**(MOCK 파이프라인 검증). 임베딩 연결 대기 |
| Evaluation & Comparison / Ablation | 다음 |
| Presentation | 마지막 |

---

## 2. 파이프라인 (Method Overview → 실제 산출물)

```
데이터 (EC-Bench: train_30/100, test_ec, price  →  나중에 자체 UniProt)
        │
        ▼  [1] extract_esm.py  — frozen ESM (650M 로컬 → 3B REBELX → ESM3)
   임베딩 .npz 저장 (ids[str], emb[float32 N×D])   ← 차원 D는 모델에 따라 다름(자동감지)
        │
        ▼  [2] heads/  — MLP · CNN · Transformer · KAN  (+ 레벨별/계층 출력)
        │
        ▼  [3] 학습  — head별 (train_30 / train_100)
        │
        ▼  [4] 평가  — per-level F1 (test_ec, price-149)  ※ 불완전 EC 마스킹
        │
        ▼  [5] 비교표  — vs DeepECT/ECPICK/CLEAN/HIT-EC  +  ablation
```

---

## 3. 단계별 상세

### [1] ESM 임베딩 추출 (가장 무거움, 1회만 · REBELX)
- 입력: 데이터 서열(train + test + price), frozen ESM.
- **모델 스케일**: `esm2_t33_650M`(D=1280, 로컬 스모크) → `esm2_t36_3B`(D=2560, **REBELX**) → `esm3`.
- **출력 규약(중요·팀 공유)**: `np.savez(path, ids=<str배열>, emb=<float32 N×D>)`.
  파일명 `{EMB_MODEL}__{split}.npz` (예: `esm2_t33_650M__train.npz`). head 노트북은 이걸 로드만 한다.
- ⚠️ **임베딩 형태가 head마다 다름**
  - **MLP / KAN** → **평균풀링 D-d** (train 258k × 1280 ≈ 1.3GB / 3B는 ≈ 2.6GB, 저장 쉬움)
  - **CNN / Transformer** → **per-residue (L×D)** 필요 → 전량 저장은 수백 GB라 비현실적
    → 학습 중 on-the-fly 계산 또는 길이 truncate(예: 512/1022) + float16
- 추천: **pooled 먼저 추출 → MLP/KAN 먼저**, CNN/Transformer는 다음 단계 per-residue.

### [2] 모델 구현 (팀 4명 → head 1개씩 분담)
- 공통 인터페이스: `pooled 임베딩 → EC 예측`, 평가는 per-level F1로 통일.
- **계층/레벨 출력**: EC 4단계(a.b.c.d)를 레벨별 예측 + 계층 일관성(부모 EC가 맞아야 자식 허용) — HIT-EC 아이디어.

### [3] 학습
- 같은 데이터·같은 평가로 head별 학습.
- **train_30(엄격/OOD)와 train_100(상한) 둘 다** 돌려 격차 확인.
- 불균형 대응: class weighting / focal / asymmetric loss / 희귀 EC long-tail.

### [4] 평가
- `test_ec`(468) + `price-149`(149)에서 **per-level F1** (macro/micro).
- **불완전 EC 마스킹**: `1.-.-.-`처럼 일부 레벨만 아는 라벨은 *아는 레벨까지만* 채점.
- **닫힌 어휘 주의**: 정답 EC가 train 어휘에 없으면 구조적 미스 → 모든 head가 같은 어휘 쓰면 비교는 공정.

### [5] 비교 & Ablation
- baseline(DeepECT/ECPICK/CLEAN/HIT-EC)은 **EC-Bench 공개 수치 인용**(재현 X, 시간 절약).
- ablation 축: head 종류 / 계층 유무 / train_30 vs 100 / 임베딩 크기(650M↔3B↔ESM3) / pooled vs per-residue.

---

## 3-K. KAN head (류도형 담당) — 상세

> 노트북: [`project3/KAN.ipynb`](project3/KAN.ipynb) (v1 완성, MOCK 파이프라인 검증 완료)

**설계 결정 (B안): 공유 ESM 임베딩 위 *KAN 분류 헤드* + per-level F1.**
- 저자 공식 repo([`datax-lab/kan_ecnumber`](https://github.com/datax-lab/kan_ecnumber))의 `CLEAN_KAN`은
  사실 **CLEAN의 대조학습(SupConHardLoss) 그대로 + MLP만 KAN으로 교체**(출력=256-d 임베딩, 추론=EC 중심거리).
  → 우리는 그걸 **그대로 쓰지 않고**, 팀의 *4-head 사과대사과 비교*에 맞춰 **KAN을 일반 분류 헤드**로 사용.
- 참고 설정 차용: KAN `grid_size=10, spline_order=3`, `efficient-kan` 구현.

**구조**: `임베딩(D) → KAN trunk[D,512] → LayerNorm → Dropout(0.1) → 레벨별 KAN 헤드 4개([512, N_L])`
- 레벨별 멀티라벨 **마스킹 BCE** (그 레벨을 아는 샘플만 손실/채점) → 불완전 EC 안전 처리.
- 출력 클래스 수(train_100, `MIN_CLASS_FREQ=1`): **L1=6, L2=66, L3=245, L4=4591**.

**교체 가능성 (계획 반영)**
- 데이터: `DATA_DIR`만 교체(EC-Bench → 자체 UniProt).
- 임베딩: 입력 차원 **자동 감지** → 650M/3B/ESM3 코드 변경 없이.
- 임베딩 추출은 노트북 밖(REBELX) → 노트북은 `.npz`만 소비.

**노트북 단계**: Config → 데이터 로드(pandas) → 레벨별 어휘/마스크 → 임베딩 로드(자동차원) →
Dataset/collate(배치서 dense 생성, 5GB 폭발 회피) → KAN 모델 → 마스킹 BCE 학습 → per-level F1 → 결과 저장 → 로드맵.

**검증 상태**: torch/efficient_kan 미설치 로컬에서 **데이터 파이프라인(파싱·어휘·인코딩·마스킹·F1) 실측 통과**.
모델 학습은 임베딩+GPU(REBELX) 연결 후.

**바로 다음**: REBELX 임베딩 추출 → `MOCK=False` 650M 실측 → train_30 vs 100 / price-149 → 3B → ESM3.

---

## 4. 진행 원칙

**"한 번에 다" ❌ → 스모크 테스트 먼저:**
> 데이터 일부(`MAX_TRAIN`) → ESM pooled → **head** → test_ec per-level F1 **숫자 1개** 뽑기
> → 파이프라인 검증 후 head·split·계층·임베딩크기로 확장.

가장 큰 관문 = **GPU 환경**. → **UNLV REBELX 확보**: **A30 GPU + conda + IP접속 Jupyter**(Slurm 아님 →
배치스크립트 불필요, Jupyter 터미널/셀에서 직접 실행). 650M은 로컬 M5(MPS)에서도 스모크 가능.

---

## 5. 데이터 (확정 + 실측 사실)

`EC-Bench_data/` — train_100(258,386) / train_30(257k, train_100의 부분집합·879개 차이) /
test_ec(468) / price-149(149) / price(183). 상세: `EC-Bench_data/README.md`.
스키마: `id,seq,ec_number`.

> train_30/100 차이는 train↔test 분리도(OOD 난이도)이지 train 크기 축소가 아님 (CLEAN split과 반대 개념).

**라벨 실측(반드시 코드에 반영):**
- **멀티라벨 존재** — `;`가 아니라 **따옴표로 감싼 쉼표**(`"2.5.1.10,2.5.1.1"`). test 26/468, train 12,511/258,386.
  → **pandas 파싱 필수**(awk/수동 split 금지).
- **불완전 EC 흔함** — `3.4.24.-`, `1.-.-.-` 등. test 깊이분포 ≈ {L1:79, L2:28, L3:207, L4:195}.
  → **레벨별 마스킹** 필요.

---

## 6. 미해결 결정 사항 (TODO)

- [x] GPU 환경 → **UNLV REBELX**(A30/conda/Jupyter-over-IP), 로컬 M5(MPS) 스모크 가능
- [x] `extract_esm.py` 작성 (`project3/extract_esm.py`; CUDA/MPS 자동, resume, .npz 규약) — *실행 대기*
- [ ] conda 환경 + 의존성(torch/fair-esm/efficient-kan) 설치 후 임베딩 추출
- [ ] `MIN_CLASS_FREQ` 값 팀 통일 (모든 head 동일 어휘 → 공정 비교)
- [ ] head 최종 목록/표기 통일 (MLP / CNN / Transformer / KAN — slide19/20)
- [ ] per-residue 임베딩 전략 (on-the-fly vs truncate+저장) — CNN/Transformer용
- [ ] 계층 손실 구체 설계 (레벨 일관성)
- [ ] baseline 공개 수치 인용 범위 확정
- [ ] (후순위) 자체 UniProt 전처리본 → `DATA_DIR` 스왑
