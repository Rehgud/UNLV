# HierEC — 계층 인식 EC 예측 (C-HMCNN on ESM-3B)

팀원의 flat 모델(MLP/CNN/Transformer)·CLEAN(contrastive)과 **같은 ESM-3B features·
similarity split·레벨별 F1** 을 쓰되, head 를 **C-HMCNN**(계층 인식)으로 바꾼 모델.

## 핵심 아이디어
EC 는 4단계 트리다. flat 모델은 각 레벨을 따로 예측해 "L4 는 맞다는데 L1 은 아니다"
같은 **계층 모순**을 낸다. C-HMCNN 의 **MCM(Max Constraint Module)** 은
`부모 확률 = 자기+자손 확률의 max` 로 만들어 **자식 <= 부모** 를 구조적으로 강제한다.
→ 모순이 사라지고, 하위 레벨(L3/L4)이 특히 개선된다. (NeurIPS 2020, Giunchiglia & Lukasiewicz)

## 파일
| 파일 | 역할 |
| --- | --- |
| `hierarchy.py` | EC 트리 구성 + MCM scatter 쌍(조상/자손) + 라벨 인코딩 |
| `model.py` | base MLP + `get_constr_out`(MCM) |
| `losses.py` | MCLoss (음성=전체 자손 max, 양성=양성 자손 max, 그 뒤 BCE) |
| `metrics.py` | 레벨별 macro/micro P/R/F1 (ECPICK 과 동일) |
| `data.py` | 데이터 로딩. `load_synthetic`(지금 테스트용) + `load_real`(★TODO★) |
| `train.py` | 학습/평가 메인 |
| `run_hier.sbatch` | 클러스터 제출 |

## 지금 바로 돌려보기 (데이터 없이)
```bash
python train.py --synthetic --epochs 100 --eval-every 20
```
랜덤 feature+라벨로 model/loss/metric 전체가 도는지 검증한다. (에러 없으면 배관 OK.)

## 데이터 오면 할 일 — `data.py` 의 `load_real` 하나만 수정
팀원의 실제 포맷에 맞춰 3개만 채우면 된다 (나머지 코드는 그대로):
1. **feature** — `features_all.pt` 가 dict{id:tensor} 인지 tensor[N,D] 인지, per-residue 면 mean-pool.
   → `train_from_esm_features.py` 의 feature 로딩부를 그대로 옮기면 가장 안전.
2. **split** — `split.json` 구조(train/test id, 또는 similarity 버킷 키).
3. **label** — id → ec_number 매핑 (csv 컬럼명 확인).

그다음:
```bash
python train.py --features .../features_all.pt --split .../split70/split.json \
                --labels .../train.csv --epochs 300 --tag split70
```
similarity 버킷별(100/70/50/30)로 `--split`/`--tag` 만 바꿔 반복 → 팀원 표와 나란히 비교.

## 주의 / TODO
- `load_real` 의 feature/split/label 포맷은 **데이터 확인 후 확정** (지금은 가정).
- 불완전 EC 마스킹은 `--mask-unknown 1`(기본)로 ECPICK 방식 적용 중.
- baseline 비교: 같은 base MLP 를 MCM 없이 돌리면(=flat) 계층 제약의 순효과를 분리해 볼 수 있음.
