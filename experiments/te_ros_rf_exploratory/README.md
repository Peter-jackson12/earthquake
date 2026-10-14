# TE·ROS 랜덤포레스트 탐색 — 로컬 인계

**분리된 연구**: T6 지역 Ablation C/R12/R1/R0와는 관계없는 개발 데이터 탐색. 기존 T6 브랜치 변경 없음.

- 결과 문서: [TE_ROS_RF_EXPLORATORY_20261014.md](../../reports/TE_ROS_RF_EXPLORATORY_20261014.md)
- 모델별·Fold별·혼동행렬·해시: [te_ros_rf_exploratory_summary.json](../../reports/te_ros_rf_exploratory_summary.json)
- 원본 제공 ZIP: earthquake_TE_ROS_development_eval_20261014.zip (코드 2 + 결과 JSON 2 + README, 비공개 데이터 없음)
- 이번 Git 기록 자체는 새 모델 학습이 아니며, 저장된 모델과 행별 OOF가 없어 **집계 산술만 확인**된 결과.
- RF 5Fold Macro F1 평균 0.700427, SD 0.001070, Fold4=0.698739, 3등급 Recall 67.30%, 3→1=744.
- Fold0에서만 LR, DT, RF, XGBoost, LightGBM 5모델 비교. RF는 전체 5Fold를 추가 수행 (Fold0 재사용).

## 로컬 참조 (기존 브랜치/작업트리 유지)

~~~powershell
git fetch origin peter-exp/te-ros-rf-exploratory-v1
git show origin/peter-exp/te-ros-rf-exploratory-v1:reports/TE_ROS_RF_EXPLORATORY_20261014.md
git show origin/peter-exp/te-ros-rf-exploratory-v1:reports/te_ros_rf_exploratory_summary.json
git worktree add --detach ..\earthquake-te-ros-review origin/peter-exp/te-ros-rf-exploratory-v1
~~~

원본 코드 실행은 소스 코드와 development.csv / split_manifest.csv를 **저장소 밖 비공개 폴더의 같은 디렉터리**에 두고 실행한다. 원본 파일은 원래 ZIP의 멤버 SHA로 검증한다. 실행 코드는 별도 experiments/te_ros_rf_exploratory/ 아래에 보존한다. 기존 Git 코드/결과/원본 데이터 및 T6 출력 디렉터리를 덮어쓰지 않는다.

기본은 **읽기·비교만** 진행하고, 재학습은 별도 지시가 있을 때 독립 로컬 재현 ID로 실행한다. 원래 실행 명령은 python evaluate_te_ros.py 후 python validate_rf_5fold.py이며, 마지막 스크립트는 이미 생성된 fold0 결과를 재사용한다.

후속 실험은 1개 요인씩 분리하고 지역 Group CV를 적용할 것. 0.7 근소 초과를 배포 성능으로 해석하지 않는다.
