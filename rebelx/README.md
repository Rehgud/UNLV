# rebelx — ECPICK & HierEC on REBELX 클러스터

팀원 전처리 데이터(`train_ec/val_ec/test_ec.csv`)로 ECPICK 과 HierEC 를 REBELX 에서 돌리는 디렉터리.

```
rebelx/
├── prep_data.py          # 원본 CSV(Entry/Sequence/EC number) -> id/seq/ec_number 표준화
├── data/                 # ★ prep_data.py 결과 (파이프라인 입력)
│   ├── train_ec.csv      # 153,219 rows | uniq EC 2,447
│   ├── val_ec.csv        #   5,583 rows
│   └── test_ec.csv       #   5,584 rows
├── ecpick/               # ▶ 지금 실행 가능
│   ├── ec_ecpick_pipeline.py
│   └── run_ecpick.sbatch
└── hierec/               # ⏸ ESM feature 대기 (STATUS.md 참고)
    ├── train.py, model.py, losses.py, hierarchy.py, metrics.py, data.py
    ├── run_hier.sbatch
    └── STATUS.md
```

## 데이터 준비 (이미 완료)
원본 `../ECPICK/data/{train,val,test}_ec.csv` 는 컬럼이 `Entry/Sequence/EC number` 인데,
ECPICK 파이프라인은 `seq/ec_number` 를 읽는다. `prep_data.py` 가 컬럼명만 표준화한다
(서열·라벨 내용은 그대로). 이미 `data/` 에 생성돼 있음. 재생성하려면:
```bash
python prep_data.py            # ../ECPICK/data 에서 읽어 data/ 에 저장
```

## ECPICK — 지금 실행 가능 ▶
**ECPICK 논문과 동일한 설정** + **고정 분할**(train_ec 학습 → test_ec 평가):
- ensemble 10, batch 32, dropout 0.8, relu 384, beta 0.6, class-weights on, epochs 20
- 서열 → one-hot → ECPICK CNN (ESM 불필요)

```bash
cd ecpick
sbatch run_ecpick.sbatch
squeue -u $USER            # 상태
tail -f ecpick_<jobid>.log # 로그
# 결과: results/ECPICK__seq__train_ec.csv (per-level macro/micro F1)
```
> val_ec.csv 는 별도 보관(논문 방식은 train→test 고정 분할). dev 세트로 쓰고 싶으면 활용 가능.

## HierEC — 보류 ⏸
ESM 을 거친 feature 가 아직 없어 실제 학습은 대기. 배관 검증(랜덤 feature)만 지금 가능.
자세한 내용과 재개 절차는 [`hierec/STATUS.md`](hierec/STATUS.md).

## 클러스터로 올리기
gpu 노드는 인터넷이 없지만 이 파이프라인은 torch/sklearn/pandas/numpy 만 쓰므로
(`ecpick` conda env 에 이미 설치됨) 다운로드가 필요 없다. 로컬에서 전송:
```bash
rsync -av --exclude '__pycache__' ~/UNLV/rebelx/  <user>@rebelx:~/DOHYEONG/rebelx/
```
sbatch 스크립트는 `$HOME/DOHYEONG/rebelx/{ecpick,hierec}` 경로를 가정한다
(다른 곳에 두면 `SLURM_SUBMIT_DIR` 로 자동 해결되거나 스크립트의 `cd` 경로 수정).
`--partition gpuq-a30` 는 `sinfo` 로 실제 GPU 파티션 확인 후 필요시 변경.
```
