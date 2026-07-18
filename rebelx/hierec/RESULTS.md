# HierEC (C-HMCNN) 결과 & 해석 — ESM-2 3B + 계층 인식 head

> 팀 ESM-2 3B feature + 팀 split으로, head를 **C-HMCNN(계층 인식)**으로 학습. REBELX gpu001, 2026-07-10 완료 (300 epoch, ~26분).

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
ep10부터 이미 L1 0.95/L4 0.68 수렴, 300까지 안정적 진동(±0.02). 300 epoch은 충분하고도 남음.

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

---

## 5. 논문 대비 정리

- **C-HMCNN 논문 그대로**: MCM(get_constr_out), MCLoss 구조
- **적응/추가**: (a) pos_weight 추가, (b) 입력 = ESM-2 3B feature(논문은 tabular), (c) 하이퍼파라미터·평가는 팀 프로젝트(ECPICK 비교)에 맞춤 — C-HMCNN 논문 값 아님

---

## 6. vs ECPICK (비교 — ECPICK 수정본 재실행 후 확정)

정성적으로 HierEC(ESM)이 ECPICK(서열 CNN)을 전 레벨에서 크게 앞섬. 정확한 나란한 비교표는 **ECPICK 세미콜론 버그 수정본 재실행이 끝나면** 추가 (기존 ECPICK 숫자는 버그본이라 교체 예정).

| Level (micro-F1) | ECPICK (서열, 버그본*) | **HierEC (ESM+계층)** |
|---|---|---|
| L1 | ~0.66 | **0.95** |
| L2 | ~0.46 | **0.90** |
| L3 | ~0.42 | **0.87** |
| L4 | ~0.49 | **0.67** |

\* ECPICK 숫자는 세미콜론 버그 수정 전 값 — 수정본으로 교체 예정.

---

## 7. 산출물 (REBELX `~/DOHYEONG/rebelx/hierec/`)

- `results/hier_chmcnn__real.csv` — 최종 per-level macro/micro P/R/F1
- 코드: `train.py`(pos_weight), `losses.py`(MCLoss+posw), `model.py`(MCM), `hierarchy.py`(세미콜론 수정), `data.py`(load_real 완성)

---

## 멘토 요약 (3문장)

C-HMCNN(계층 인식 head)을 팀의 ESM-2 3B feature·split으로 학습해, L1 micro-F1 0.95 · L4 0.67의 강한 성능을 얻었다. 논문의 MCLoss는 그대로면 L4가 0으로 붕괴해서, EC의 극단적 불균형을 상쇄하는 pos_weight를 추가해 L4를 0→0.67로 살렸다. 서열 CNN(ECPICK)보다 전 레벨에서 크게 앞서, "ESM feature + 계층 인식 head" 접근의 유효성을 보여준다.
