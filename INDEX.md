# UNLV — enzyme EC-prediction 프로젝트 구조

정리일: 2026-07-14 · 리포트에 필요한 것만 남기고 중복·백업·캐시 제거함.

## 폴더 안내

| 폴더/파일 | 내용 |
|---|---|
| **report/** | 제출용 결과 보고서 + 그림. `EC예측_결과보고서_0714.md`(메인), `figs/`(fig1~3 png+svg), `ECPICK_결과_해석.md`·`HierEC_결과_해석.md`(상세 해석) |
| **results_0714/** | 실제 실험 결과 스냅샷(재현 기록). `ecpick/`·`hierec/` 각각 코드·로그·results CSV/PNG/npz. REBELX 2026-07-14 런 |
| **rebelx/** | 유지보수용 코드 본체. `ecpick/`(서열 CNN), `hierec/`(C-HMCNN), `data/`(팀 전처리 train/val/test_ec.csv), `RESULTS.md`·`STATUS.md` |
| **ECPICK/data/** | 원본 데이터셋(EC-Bench_data, new_preprocessed, train/val/test_ec.csv) — 재실험용 보관 |
| **Study/** | 강의 자료 PDF |
| **HW/** | 과제 HW1~4 |
| **EXPERIMENT_PLAN.md** | 실험 계획 |
| **주간보고서_..._3주차_영문병기.docx** | 주간보고서 |

## 핵심 결과 (C-HMCNN, EC-Bench test 5,321)

micro-F1: L1 0.947 · L2 0.904 · L3 0.866 · L4 0.669 (macro는 롱테일 때문에 L4 0.317).
자세한 해석은 `report/` 참고.
