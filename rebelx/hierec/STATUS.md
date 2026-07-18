# HierEC — 현재 상태: ⏸ ESM feature 대기 중 (보류)

## 왜 지금 못 돌리나
HierEC(C-HMCNN)은 **원시 서열이 아니라 ESM-3B feature**(`features_all.pt`, dim 2560)를
입력으로 받도록 설계돼 있다 (`data.py`의 `load_real`, `train.py --features ...`).

팀원이 준 전처리 데이터(`../data/{train,val,test}_ec.csv`)는 **원시 서열**이라
아직 ESM 을 거친 feature 가 없다 → HierEC 실제 학습은 **보류**.

## 지금 할 수 있는 것
- 파이프라인(model/loss/metric) 배관 검증만 랜덤 feature 로 가능:
  ```bash
  python train.py --synthetic --epochs 100 --eval-every 20
  ```
  (에러 없이 돌면 코드는 정상. 실제 성능과는 무관.)

## ESM feature 가 준비되면 (3단계)
1. **feature 추출** — `../data/*.csv` 의 서열을 ESM-2/3B 로 임베딩해서
   `features_all.pt`(dict{id: Tensor} 또는 Tensor[N,D]) 생성. per-residue 면 mean-pool.
2. **`data.py` 의 `load_real` 확정** — feature/split/label 3개 매핑만 실제 포맷에 맞춤:
   - feature: `features_all.pt` 로딩부
   - split: train/test id (또는 similarity 버킷 split.json)
   - label: `../data/train_ec.csv` 의 `id -> ec_number`
3. **학습 실행** — `run_hier.sbatch` 의 (2) 블록 주석 해제 후 `sbatch run_hier.sbatch`

자세한 배경은 `README.md` 참고.
