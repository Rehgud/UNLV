# HierEC (C-HMCNN) 결과 & 해석 — ESM-2 3B + 계층 인식 head

> 팀 ESM-2 3B feature + 팀 split으로, head를 **C-HMCNN(계층 인식)**으로 학습. REBELX gpu001, 2026-07-10 완료 (300 epoch, ~26분, job 119279).

📊 그래프: `HierEC_perlevel.png` (레벨별 P/R/F1), `ECPICK_vs_HierEC.png` (모델 비교)

---

## 0. 한 줄 요약

**ESM-2 3B feature + 계층 제약(MCM) + pos_weight 조합이 전 레벨에서 강력한 성능**을 냈다 (L1 micro 0.95 → L4 micro 0.67). 서열 CNN(ECPICK)보다 크게 앞서며, "ESM + 가벼운 head" 접근이 통한다는 강한 증거.

---

## 1. 세팅

- **모델**: C-HMCNN (Giunchiglia & Lukasiewicz, NeurIPS 2020) — base MLP → **MCM(Max Constraint Module)**로 "자식 확률 ≤ 부모 확률" 구조 강제 → **MCLoss**
- **입력**: `features_all.pt`의 ESM-2 3B mean-pooled 임베딩 `[N, 2560]` (별도 라벨 CSV 불필요 — targets/labels 내장)
- **데이터**: 팀 split.json. **train 147,395 / test 5,321** (valid 5,350은 미사용, train→test 평가)
- **계층 노드**: 2,149개 (L1=7, L2=67, L3=211, L4=1,864)
- **하이퍼파라미터**: hidden 1024, layers 2, dropout 0.3, batch 512, Adam lr 1e-3, wd 1e-4, epochs 300, threshold 0.5
- **pos_weight ON** (cap 50) ← ★ 아래 참고

---

## 2. 최종 성능 (test 5,321)

| Level | macroP | macroR | macro-F1 | microP | microR | micro-F1 |
|---|---|---|---|---|---|---|
| L1 | 0.896 | 0.982 | **0.936** | 0.913 | 0.985 | **0.947** |
| L2 | 0.674 | 0.875 | 0.744 | 0.838 | 0.982 | 0.904 |
| L3 | 0.600 | 0.783 | 0.656 | 0.778 | 0.978 | 0.866 |
| L4 | 0.292 | 0.401 | **0.317** | 0.549 | 0.858 | **0.669** |

---

## 3. 핵심 해석

**1. 전 레벨 고성능, 특히 recall이 높음**
micro-recall이 L1~L3에서 0.98, L4도 0.86. pos_weight로 양성을 적극 예측 + MCM이 계층 일관성을 유지해 하위 레벨까지 잘 잡는다.

**2. macro-F1도 크게 개선 (희귀 클래스 포함)**
L4 macro-F1 **0.317** — 희귀 클래스가 몰린 L4에서도 서열 CNN 대비 큰 개선. ESM 임베딩이 서열 one-hot보다 정보량이 훨씬 많아 저빈도 클래스 구분에 유리.

**3. 수렴이 빠르고 안정적**
ep10부터 이미 L1 0.95/L4 0.68 수렴, 300까지 안정적 진동(±0.02). 300 epoch은 충분하고도 남음. (📊 `HierEC_learncurve.png`)

**4. L4는 여전히 가장 어려움**
L4 macro 0.317 / micro 0.669 — 상위 레벨(0.9+)보다 낮다. 세부 4자리 기능은 본질적으로 어렵지만, ECPICK 대비 훨씬 나음.

---

## 4. ★ pos_weight — 논문에서 벗어난 핵심 수정

원래 C-HMCNN의 MCLoss는 **pos_weight가 없다.** 그대로 돌리면 **L4가 0.000으로 붕괴**(희귀 노드가 threshold 0.5를 못 넘음 = all-zero-F1 함정). ECPICK과 동일하게 **노드별 pos_weight(neg/pos, cap 50)**를 MCLoss에 넣어 해결.

| L4 micro-F1 | pos_weight 없음(논문 그대로) | **pos_weight 추가** |
|---|---|---|
| L4 | **0.000** | **0.669** |
| L1 | 0.72 | 0.95 |

→ 멘토 보고 시 이 대비가 "왜 pos_weight를 넣었는지"의 근거. **논문 방법론(MCM/MCLoss)은 충실 재현, pos_weight만 EC 불균형 때문에 추가.**

**★ 왜 L1(0.72)은 버티고 L4(0.000)만 완전히 죽는가 — 추정 설명(★확인된 사실 아님, 발표 시 구분해서 말할 것):**
C-HMCNN의 MCM은 각 노드의 최종 출력을 "자기 자신의 원점수 vs 그 아래 모든 자손 노드 원점수 중 최댓값"으로 계산해 "부모 확률 ≥ 자식 확률"을 구조적으로 강제한다. 이 구조 때문에:
- **L1(최상위)**: 자손이 수백~수천 개라, 그중 하나라도 확률이 높게 나오면 그 신호를 그대로 물려받아 최종 점수가 끌어올려짐 → pos_weight 없이도 어느 정도 버팀(0.72)
- **L4(리프 노드, 자손 없음)**: 물려받을 자손이 없어 순전히 자기 자신의 raw 점수에만 의존 → 불균형 문제를 완충 장치 없이 그대로 맞아 0.000

이건 MCM 구조에 근거한 **팀의 해석**이며, 별도 ablation(예: MCM 없이 같은 불균형 조건에서 L1이 어떻게 나오는지)으로 직접 검증한 것은 아니다. Q&A 등에서 이 설명을 쓸 때는 "확인된 사실(L4가 0으로 붕괴)"과 "추정 설명(왜 L1은 안 죽는지)"을 구분해서 말할 것.

---

## 4.5 ★ 학습개수별 성능 (빈도분석) — ESM이 희귀 클래스를 살린다

ECPICK과 동일한 방식(클래스별 학습개수 구간 × recall)으로 분석. 📊 `HierEC_frequency.png`, `ECPICK_vs_HierEC_frequency.png`

**L4 (저빈도 집중), 학습개수별 micro-recall — ECPICK 대비:**

| 학습개수 | C-HMCNN recall | ECPICK recall | C-HMCNN 맞춘 클래스 비율 |
|---|---|---|---|
| 6-10 | 0.08 | 0.00 | 8.8% |
| 11-20 | 0.19 | 0.00 | 20.6% |
| 21-50 | **0.56** | 0.00 | 59.6% |
| 51-100 | **0.84** | 0.03 | 92.4% |
| 101-500 | 0.98 | 0.65 | 99.4% |
| 501+ | 0.99 | 0.88 | 100% |

**핵심 발견: ESM 임베딩이 "학습에 필요한 예시 개수"를 약 4~5배 낮춘다.**
- ECPICK(서열 CNN)은 **~100개 이상** 있어야 학습되고 50개 이하는 recall 0
- C-HMCNN(ESM)는 **21~50개에서 이미 recall 0.56**, 51~100개면 0.84로 강함
- 즉 ESM-2 3B의 진화적 정보 덕에 **few-shot에 가까운 희귀 클래스 학습**이 가능. 이것이 L4 macro-F1이 ECPICK 0.14 → C-HMCNN 0.32로 오른 근본 원인
- 단, **10개 미만**은 C-HMCNN도 거의 실패(1-5개 recall 0, 6-10개 0.08) — 극단적 희귀는 여전히 한계

---

## 5. 논문 대비 정리

- **C-HMCNN 논문 그대로**: MCM(get_constr_out), MCLoss 구조
- **적응/추가**: (a) pos_weight 추가, (b) 입력 = ESM-2 3B feature(논문은 tabular), (c) 하이퍼파라미터·평가는 팀 프로젝트(ECPICK 비교)에 맞춤 — C-HMCNN 논문 값 아님
- **빈도분석 코드**: `frequency.py`(신규) + `train.py` 패치로 ECPICK과 동일 스키마 CSV 생성

---

## 6. vs ECPICK (비교 — ECPICK 수정본 재실행 완료, 확정)

C-HMCNN(ESM+계층)이 ECPICK(서열 CNN)을 **전 레벨에서 크게 앞섬.** 서열 one-hot CNN은 원거리 의존성·진화 정보를 못 잡아 세부 EC(L3/L4)에서 격차가 특히 벌어진다. 📊 `ECPICK_vs_HierEC.png`

| Level | ECPICK micro-F1 | **C-HMCNN micro-F1** | ECPICK macro-F1 | **C-HMCNN macro-F1** |
|---|---|---|---|---|
| L1 | 0.685 | **0.947** | 0.660 | **0.936** |
| L2 | 0.501 | **0.904** | 0.404 | **0.744** |
| L3 | 0.473 | **0.866** | 0.307 | **0.656** |
| L4 | 0.529 | **0.669** | 0.142 | **0.317** |

> * ECPICK는 세미콜론 버그 수정본(job 119278, test 5,584) 확정값. 테스트셋 규모가 약간 다름(C-HMCNN n=5,321) — 동일 팀 split, 라벨 처리 차이. 방향성 비교용.
> * 격차가 가장 큰 곳은 **L3 macro-F1(0.31→0.66, 2배 이상)**·**L2 micro-F1(0.50→0.90)** — 계층 제약 + ESM 임베딩이 하위 레벨을 크게 살림.

---

## 7. 산출물 (REBELX `~/DOHYEONG/rebelx/hierec/`)

- 그래프: `HierEC_perlevel.png` (레벨별 P·R·F1), `HierEC_learncurve.png` (300 epoch test micro-F1), `HierEC_frequency.png` (학습개수별 recall), `ECPICK_vs_HierEC.png`·`ECPICK_vs_HierEC_frequency.png` (비교)
- `results/hier_chmcnn__real.csv` — 최종 per-level macro/micro P/R/F1
- `results/freq_summary__hier_chmcnn__real.csv` / `perclass__hier_chmcnn__real.csv` — 클래스별·빈도구간별 (job 119531)
- `frequency.py` (신규) + `train.py` 패치 — 빈도분석 저장 코드
- 코드: `train.py`(pos_weight), `losses.py`(MCLoss+posw), `model.py`(MCM), `hierarchy.py`(세미콜론 수정), `data.py`(load_real 완성)

---

## 멘토 요약 (3문장)

C-HMCNN(계층 인식 head)을 팀의 ESM-2 3B feature·split으로 학습해, L1 micro-F1 0.95 · L4 0.67의 강한 성능을 얻었다. 논문의 MCLoss는 그대로면 L4가 0으로 붕괴해서, EC의 극단적 불균형을 상쇄하는 pos_weight를 추가해 L4를 0→0.67로 살렸다. 서열 CNN(ECPICK)보다 전 레벨에서 크게 앞설 뿐 아니라, **빈도분석 결과 ESM이 학습에 필요한 예시 개수를 4~5배 낮춰**(ECPICK은 ~100개 필요, C-HMCNN는 21~50개면 recall 0.56) 희귀 EC 클래스를 few-shot에 가깝게 학습한다 — "ESM feature + 계층 인식 head" 접근이 long-tail 문제에 특히 유효함을 보여준다.
