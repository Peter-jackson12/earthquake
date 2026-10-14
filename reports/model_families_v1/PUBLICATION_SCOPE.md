# 공개 집계 범위

이 폴더는 model_families_v1의 공개용 요약이다. 원시 데이터·행별 OOF·모델은 없다.

`verification.json`의 aggregate_sha256은 최초 전체감사 출력의 해시 목록이다. 그중 이 폴더에는 핵심 CSV만 같은 바이트로 보존했다.
`error_profiles_compact.csv`는 전체 오류프로필의 9개 필드·6자리 반올림 표시본, `primary_confusion_matrices.json`은 전체/995/비995의 18개 CM이다.
전체 비반올림 감사 CSV, 지역별 모든 CM, 20개 모델 replay 기록은 코드·증거 전달 ZIP의 `full_audit/`에서 확인한다.

추가 검증: `aggregate_verification.json`. 실행·테스트·소스 근거: `execution_summary.json`.
실험의 상태·수치·한계는 상위 `MODEL_FAMILY_COMPARISON_20261014.md`를 우선한다.
