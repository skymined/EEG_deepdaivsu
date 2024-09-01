# 프로젝트 실행 가이드

프로젝트 실행하기 위해서 필요한 것들을 정리했습니다. 각 단계에서 필요한 디렉토리 및 파일이 무엇이 있는 지 중점적으로 봐주세요.

## 실행 순서

1. **`make_feature.py`**: 데이터 전처리 및 텐서 파일 생성
2. **`main.py`**: 모델 학습 및 테스트

## `make_feature.py` 실행 전 요구 사항

- **한줄 요약**
  - "ExtractedFeatures", "SEED_data"라는 폴더 만들고, ExtractedFeatures 폴더 내부에는 피험자 데이터 넣어주기.

- **구체적인 내용**:
  - `ExtractedFeatures/`
    - 1번부터 15번까지의 피험자 데이터가 포함되어야 하며, 각 피험자에 대해 3회의 실험 데이터가 있으므로 총 45개의 `.mat` 파일로 저장.
    - 파일 예시 : `ExtractedFeatures/1_20131027.mat`
  - `SEED_data/`
    - 새로 생성해야하는 빈 디렉토리.
    - `make_feature.py` 실행 후 `train{피험자번호}de.pt` 및 `test{피험자번호}de.pt` 파일이 저장됨.
   

## `main.py` 실행 전 요구 사항

- **한줄 요약**
  - "model", "best"라는 폴더 만들기.

- **구체적인 내용**:
  - `model/`
    - 새로 생성해야하는 빈 디렉토리.
    - `main.py` 실행 후 `test{피험자번호}_best.pt` 파일이 저장됨.
   

  - `best/`
    - 새로 생성해야하는 빈 디렉토리.
    - LOSO 진행하며 얻은 best_accuracy와 test_accuracy가 저장됨.
