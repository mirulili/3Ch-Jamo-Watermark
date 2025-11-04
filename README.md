# Jamo-based Watermark Project

## 프로젝트 구조
```
Capstone25/
├─ src/
│  ├─ __init__.py
│  ├─ main.py                    # 워터마크 생성 및 검출 파이프라인 실행
│  │
│  ├─ model/                     # 언어 모델 관련 모듈
│  │  ├─ __init__.py
│  │  ├─ load_model.py           # 모델 및 토크나이저 로딩
│  │  └─ generate.py             # 텍스트 생성 로직
│  │
│  ├─ watermark/                 # 워터마킹 핵심 로직 모듈
│  │  ├─ __init__.py
│  │  ├─ jamo_utils.py           # 한글 자모 분해 유틸리티
│  │  ├─ payload_mgr.py          # 메시지 <-> 비트열 변환 관리
│  │  ├─ hash_policy.py          # 자모 기반 해시 계산 정책
│  │  ├─ processor.py            # JamoWatermarkProcessor (워터마크 삽입)
│  │  └─ detector.py             # JamoWatermarkDetector (워터마크 검출)
│  │
│  └─ evaluation/                # 성능 평가 관련 모듈
│     ├─ __init__.py
│     ├─ eval_quality.py         # (예정) 생성 품질(PPL 등) 측정
│     └─ eval_robustness.py      # (현재 비활성) 강건성 테스트
│
├─ .gitignore                    # Git 추적 제외 설정
├─ Makefile                      # 빌드 및 실행 자동화
├─ README.md                     # 프로젝트 설명
└─ requirements.txt              # 의존성 라이브러리 목록
```

## 실행 방법
1.  **의존성 설치**:
    ```bash
    make install
    ```
2.  **프로그램 실행**:
    ```bash
    make run
    ```
    `src/main.py`가 실행되며, 워터마크를 삽입하여 텍스트를 생성하고, 생성된 텍스트에서 다시 메시지를 복원하는 전체 과정을 수행합니다.

## 핵심 동작 원리
1.  **자모 채널 분리**: 한글 음절을 초성, 중성, 종성 3개의 채널로 분리하여 각 채널에 독립적으로 워터마크 비트를 할당합니다.
2.  **해시 기반 매칭**: 각 토큰의 자모 인덱스로부터 해시 값을 계산하고, 이 값이 해당 스텝의 목표 비트와 일치하는지 확인합니다.
3.  **조건부 스텝 동기화**:
    *   **삽입(Processor)**: 로짓(확률)에 편향을 적용한 후, **가장 유력한 후보 토큰**이 목표 비트와 일치할 경우에만 워터마크가 삽입된 것으로 간주하고 다음 비트로 넘어갑니다 (`step_t` 증가).
    *   **검출(Detector)**: 생성된 텍스트의 토큰을 순서대로 읽으며, 해당 토큰의 해시 값이 **현재 찾아야 할 목표 비트**와 일치할 경우에만 워터마크를 추출하고 다음 비트로 넘어갑니다 (`step_t` 증가).
    *   이 방식을 통해 샘플링의 불확실성 속에서도 생성기와 검출기가 동일한 규칙으로 스텝을 진행하여 동기화를 유지합니다.