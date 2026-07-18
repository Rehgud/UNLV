# Q&A 예상 질문 대비 — ECPICK / C-HMCNN 파트

**발표자:** 도형 (Dohyeong Ryu) · **대상:** Mingon Kang 교수님 Q&A (5분)
**전제:** 교수님은 ECPICK 원논문(Han et al., *Briefings in Bioinformatics* 2024)을 낸 랩 소속이라, 방법론적 rigor를 깊게 파고들 가능성이 높음.

---

## 0. 가장 확실한 질문 — "pos_weight 왜 켰어?"

이 프로젝트 파트의 핵심 질문. 완성된 답변을 그대로 말할 수 있게 준비.

### English (easy, ~30 sec)

> We turned pos_weight on because without it, both models completely failed on the hardest level. The reason is extreme class imbalance — ECPICK has 1,889 classes at L4, most with very few training examples. Without pos_weight, the model's safest bet is to predict "negative" for everything, so the output probability never even gets close to the threshold. We actually tested our mentor's exact setting first — no pos_weight, threshold 0.19 — and confirmed this: L2 through L4 all collapsed to exactly zero. Adding pos_weight fixes this by giving extra loss-weight to rare positive labels, pushing the model to actually try predicting them. The trade-off is precision goes down — more false alarms — but without it, the model doesn't work at all for the classes that matter most.

### 한국어

> pos_weight를 켠 이유는, 안 켜면 두 모델 다 가장 어려운 레벨에서 완전히 실패하기 때문이에요. 원인은 극심한 클래스 불균형이에요 — ECPICK은 L4에서 클래스가 1,889개인데, 대부분 학습 예시가 아주 적어요. pos_weight 없이는 모델 입장에서 제일 안전한 선택이 "전부 음성"으로 찍는 거라서, 출력 확률 자체가 threshold 근처도 못 가요. 저희가 실제로 멘토님이 지정하신 설정(pos_weight 없음, threshold 0.19)을 먼저 테스트해봤는데, L2부터 L4까지 전부 정확히 0으로 무너지는 걸 확인했어요. pos_weight를 추가하면 희귀한 양성 라벨에 더 큰 손실 가중치를 줘서, 모델이 실제로 그 클래스들을 예측해보게 만들어요. 대신 precision은 떨어져요 — 헛방이 늘어나거든요. 근데 이게 없으면 제일 중요한 클래스들에서 모델이 아예 작동을 안 해요.

**핵심 논리 3단계:**
1. 안 켜면 → 붕괴 (실측으로 확인함, 멘토 설정으로도 검증함)
2. 왜 붕괴하는가 → 극심한 불균형 + 확률이 threshold를 못 넘음
3. 켜면 → recall은 살지만 precision은 희생 (트레이드오프를 숨기지 않고 인정)

---

## 1. 방법론 / Rigor 관련 — 가장 파고들 가능성 높음

### Q1. threshold 0.19 재현했을 때 왜 붕괴했다고 생각해? threshold 문제 아니야?
저장된 확률을 0.01~0.95 전 구간 스윕해봤는데도 회복 안 됨. 진짜 원인은 pos_weight 없이는 출력 확률 자체가 0.19는커녕 그 아래로 다 깔려서(L2 최대 0.16). 즉 threshold가 아니라 확률 분포 자체의 문제.

### Q2. pos_weight ON 기준 "최적 threshold" 표는 test set으로 찾은 거 아니야? 그거 data leakage 아니야?
맞음 — 이미 그렇게 캐비어트를 달아놨음. "이 정도 개선 여지가 있다"는 상한선 참고용이고, 정직한 검증은 val_ec.csv로 해야 하는데 아직 안 했다고 솔직히 답할 것.

### Q3. ECPICK이랑 C-HMCNN 테스트셋 크기가 다른데(5,584 vs 5,321) 공정한 비교야?
같은 팀 split·라벨인데 후처리 차이로 약간 다름. 격차가 워낙 커서 순위엔 영향 없지만, 엄밀한 동일 비교는 아니라고 인정.

### Q4. 교수님 랩 표준 평가방식인 10x repeated stratified hold-out(HIT-EC 논문 방식) 왜 안 썼어?
⚠️ **가장 아플 수 있는 질문.** 시간 문제로 이번엔 고정 분할 하나만 했고, rsho는 다음 단계로 남겨뒀다고 솔직히 답하는 게 나음.

---

## 2. 해석 / 설명 관련

### Q5. C-HMCNN에서 pos_weight 없을 때 왜 L1(0.72)은 안 죽고 L4만 죽었어?
MCM 구조상 부모 노드는 자손 중 최댓값을 물려받아서 버티고, 리프(L4)는 그런 완충장치가 없다는 설명. **단, 이건 우리 추정이지 논문에 직접 명시된 확인 사실은 아니라고 밝힐 것.**

### Q6. ESM이 예시를 덜 필요로 한다는 게 진짜 임베딩 때문이야, 아니면 다른 요인(구조·학습 epoch 차이)일 수도 있지 않아?
맞는 지적이라 인정. "가장 유력한 설명이지만 두 모델 구조 자체가 다르니 confound 가능성은 배제 못 한다"고 답하는 게 안전.

---

## 3. Novelty / 기여 관련

### Q7. 기존 논문 코드 그대로 돌린 거 아니야? 새로운 게 뭐야?
- ECPICK 재현 중 세미콜론 multi-label 버그 직접 발견·수정
- C-HMCNN(NeurIPS 논문, EC 태스크엔 원래 안 쓰임)을 ESM-2 임베딩에 처음 적용
- 멘토 지정 설정을 실제로 돌려서 붕괴를 실측 검증
- few-shot 격차(4~5배)를 빈도분석으로 정량화

---

## 4. 향후 계획

### Q8. 다음엔 뭐 할 거야?
7/15에 교수님·팀원들과 이미 논의한 내용 그대로: 단백질 구조(structure) 정보까지 포함해서 확장하고, 논문화 가능성 검토 중.

---

## 요약 — 가장 위험한 질문 2개

| 순위 | 질문 | 대응 |
|---|---|---|
| ⚠️⚠️ | Q4. 왜 10x rsho 안 썼나 | 시간 문제로 못 함 — 다음 단계로 인정 |
| ⚠️ | Q2. test-set leakage | 이미 캐비어트로 인지하고 있음 — val 검증 필요하다고 인정 |

두 질문 모두 **"이미 알고 있고, 한계로 인지하고 있다"**고 솔직히 말하는 게 방어적으로 가장 안전함.
