# 한국어 구음장애 개인화 음성인식

동국대학교 공개소프트웨어 프로젝트 · 컴퓨터공학전공 서지우

구음장애 사용자의 음성을 텍스트로 변환하고, 소량의 사용자 음성으로 모델을 개인화했을 때의 인식 정확도와 학습 비용을 비교하는 프로젝트입니다.

Chung et al.의 한국어 구음장애 Whisper 연구를 재현 대상으로 삼습니다.
논문 코드와 학습 모델의 출처·리비전·사용 조건은 [연구 기반 기록](docs/research-base.md)에 정리했습니다.
현재 구현은 AI Hub 샘플 점검과 기본 Whisper/논문 통합 학습 모델의 동일 구간 추론 비교입니다.
아직 개인화 학습이나 논문 성능 재현을 완료한 단계는 아닙니다.

## 현재 진행 상태

| 작업 | 상태 |
| --- | --- |
| AI Hub WAV/JSON 짝·누락·중복 확인 | 구현·실제 샘플 검증 |
| 실제 음성 헤더 확인, mono/16kHz 변환 | 구현·실제 샘플 검증 |
| 기본 Whisper / 선행연구 모델 비교 | 샘플 4개, 두 모델 총 8회 추론 |
| 출력·추론 시간·조건 저장 | 구현 |
| 파일 전체 정답을 이용한 CER/WER 평가 | 4개 파일, 처리 조건 비교 포함 20회 완료 |
| 발화별 음성·정답 정렬 | 다음 작업 |
| LoRA·Adapter 기반 개인화 학습 | 계획 |
| 사용자 서비스 화면 | 계획 |

## 처음 시작하기

### 1. 저장소와 실행 환경

아래 명령은 macOS/Linux 기준입니다. 현재 검증 환경은 macOS Apple Silicon, Python 3.12입니다. Windows에서는 가상환경의 `.venv/bin/python`을 `.venv\Scripts\python.exe`로, `.venv/bin/hf`를 `.venv\Scripts\hf.exe`로 바꿉니다. Windows 실행은 아직 검증하지 않았습니다.

Python 3.12, CPU int8에서 확인했습니다. 데이터와 모델은 저장소 밖에 둡니다.
AI Hub 데이터는 각 팀원이 접근 권한을 확보해 내려받아야 합니다.

```sh
git clone https://github.com/wooseo1/OSS-Project.git
cd OSS-Project
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-convert.txt
```

### 2. 데이터 보관

AI Hub에서 접근 권한을 확보하고 원천 WAV와 라벨 JSON을 모두 내려받아 압축을 풉니다. 내부 폴더 이름은 달라도 됩니다.

```text
작업폴더/
├── OSS-Project/              # 팀이 함께 관리하는 코드
├── datasets/
│   └── aihub-dysarthria/     # 각자 내려받은 데이터
│       └── New_Sample/
│           ├── 원천데이터/…/*.wav
│           └── 라벨링데이터/…/*.json
└── models/                  # 각자 내려받은 모델
```

AI Hub 원본은 GitHub에 올리지 않습니다. 각자 데이터를 내려받거나, 이용 조건에 맞는 공동 접근 환경을 사용합니다. 음성·라벨·환자 정보·모델·실험 출력은 코드와 분리해 보관합니다.

### 3. 모델 준비

아래 `../models`는 로컬 보관 폴더입니다. 개인 경로에 맞게 바꿔도 됩니다. 모델 버전을 고정해 팀원이 같은 파일을 사용하도록 합니다. 최초 다운로드에는 인터넷 연결과 여유 디스크 공간이 필요합니다.

```sh
.venv/bin/hf download Systran/faster-whisper-small \
  --revision 536b0662742c02347bc0e980a01041f333bce120 \
  --local-dir ../models/whisper-small-ct2

.venv/bin/hf download yoona-J/ASR_Whisper_Disease_General \
  --revision 304fe2952840b5b96ca3d4399a43d024912d4096 \
  --include '*.json' '*.txt' '*.safetensors' '*.md' \
  --local-dir ../models/whisper-dysarthria-general

.venv/bin/python scripts/convert_model.py \
  --source ../models/whisper-dysarthria-general \
  --output ../models/whisper-dysarthria-general-ct2
```

모델 변환은 처음 한 번만 필요합니다. 출력 폴더가 이미 있으면 새 폴더를 지정하세요.
변환 후 추론만 하는 팀원은 `requirements.txt`만 설치하면 됩니다.

### 4. 데이터 점검 및 비교 실행

원천 WAV와 라벨 JSON을 모두 포함하는 상위 폴더를 `--data-root`로 지정합니다. 아래 첫 명령은 모델 없이 데이터만 점검하므로 모델 다운로드 전에도 실행할 수 있습니다.
파일 순서 대신 JSON의 `File_id`로 짝을 찾으며 누락과 중복이 있으면 중단합니다.

```sh
.venv/bin/python main.py \
  --data-root ../datasets/aihub-dysarthria \
  --output-dir outputs/sample-smoke

.venv/bin/python main.py \
  --data-root ../datasets/aihub-dysarthria \
  --output-dir outputs/sample-smoke \
  --baseline ../models/whisper-small-ct2 \
  --adapted ../models/whisper-dysarthria-general-ct2 \
  --start-seconds 30 --seconds 30
```

정상 점검 시 `Validated … WAV/JSON pairs`가 표시됩니다. 현재 받은 샘플은 4쌍입니다.

`adapted`는 선행연구의 **질환 통합 학습 모델**이며 우리 팀이 개인화한 모델이라는 뜻은 아닙니다. 현재 샘플은 시작 부분에 긴 무음이 있어 30~60초 구간을 비교합니다. 다른 음성에서는 구간을 조정하세요. 구간 길이는 최대 30초입니다.
실제 WAV 헤더를 기준으로 mono/16kHz로 변환합니다.
| 생성 파일 | 내용 |
| --- | --- |
| `outputs/sample-smoke/inventory.json` | 파일 짝·길이·실제 샘플링 주파수·라벨 표기 |
| `outputs/sample-smoke/comparison.json` | 모델별 출력·구간·추론 시간·실행 조건 |
| `outputs/sample-smoke/comparison.md` | 사람이 읽는 비교 결과 |

`outputs/`는 Git 관리에서 제외됩니다. 같은 출력 경로로 재실행하면 이전 결과를 덮어쓰므로 실험을 보존하려면 새 폴더를 지정하세요.

## 평가 범위

샘플의 정답은 파일 전체 Transcript이며 발화 타임스탬프가 없습니다.
따라서 위의 30초 추론 결과에 파일 전체 정답을 붙여 CER/WER를 계산하지 않습니다.
짧은 구간 비교는 두 모델이 실행되는지 확인하는 단계입니다.
별도의 [파일 전체 평가](docs/evaluation.md)는 전체 WAV를 입력하고 전체 정답과 비교해 CER/WER를 계산합니다.
원 연구 학습 데이터와의 중복 여부도 미확인입니다.
개인화 모델의 정량 평가는 발화별 음성·정답 정렬과 적절한 화자/문장 분할을 확보한 뒤 추가합니다.

## 파일 전체 실험

첫 실험은 [2026-10-11 샘플 평가 결과](docs/experiments/2026-10-11-sample-evaluation.md)에 정리했습니다.
무음 제거 후 묶음 처리에서 기본 Whisper의 통합 CER은 15.63%, 선행 모델은 4.36%였습니다.
VAD 구간별 독립 처리에서는 각각 17.60%, 31.52%로 악화됐습니다.
단일 화자 표기의 샘플 4개에 대한 관측값이며 일반적인 성능 개선이나 개인화 효과를 뜻하지 않습니다.

```sh
.venv/bin/python -m pip install -r requirements-eval.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/evaluate_full.py \
  --data-root ../datasets/aihub-dysarthria \
  --baseline ../models/whisper-small-ct2 \
  --adapted ../models/whisper-dysarthria-general-ct2 \
  --output-dir outputs/full-eval \
  --no-vad-first-file
```

새 출력 폴더를 지정해야 합니다. 전체 파일 평가, 발화별 독립 추론, 무음 제거 유무 비교와 지표 정의는 [평가 방법](docs/evaluation.md)에 있습니다. 이 실험은 기존 모델의 탐색 평가이며 독립 테스트나 개인화 학습 결과가 아닙니다.

## 다음 개발 순서

1. 발화별 시작·종료 시점과 정답 텍스트를 연결하고 사람이 검수합니다.
2. 평가용 데이터와 학습용 데이터를 분리하고, 같은 조건에서 CER/WER를 측정합니다.
3. 사용자별 음성량을 달리해 LoRA·Adapter 개인화 실험을 진행합니다.
4. 정확도 개선폭, 학습 시간, 메모리와 추가 파라미터 수를 함께 비교합니다.

## 협업과 커밋

**한 가지 목적의 변경을 한 커밋으로** 기록합니다. 기능 개발은 작업 브랜치에서 진행하고 실행 확인 후 PR로 검토합니다. 공유한 브랜치의 기록을 강제로 덮어쓰지 않습니다.

시작·저장·공유 절차와 커밋 예시는 [협업 안내](CONTRIBUTING.md)에 있습니다. Codex에 “변경 확인하고 커밋해 줘”라고 요청하면 변경 내용·검증 결과를 확인한 뒤 작업 단위에 맞게 정리할 수 있습니다.

## 코드 구성과 출처

```text
main.py                      # 실행 진입점
scripts/sample_compare.py    # 데이터 점검과 동일 구간 추론
scripts/convert_model.py     # 학습 모델을 CPU 추론 형식으로 변환
scripts/evaluate_full.py     # 전체 파일 정답으로 CER/WER와 처리 조건 비교
requirements.txt             # 추론 의존성
requirements-convert.txt     # 변환 포함 의존성
requirements-eval.txt        # 지표 계산 포함 의존성
tests/test_evaluation.py     # 오류 개수·합산·발화 연결 검증
docs/evaluation.md           # 전체 파일 평가 방법
docs/research-base.md         # 선행연구·모델 버전·검증 기록
CONTRIBUTING.md               # 팀 협업과 커밋 안내
```

재현 대상은 [Diagnosis-aware multitask fine-tuning of Whisper for dysarthric speech recognition의 공개 저장소](https://github.com/yoona-J/Diagnosis-Aware_Multitask_Fine-Tuning_of_Whisper_for_DSR)입니다. 원본 코드 저장소에는 명시적 라이선스 파일이 없어 별도 참고 클론으로 보관하고, 이 저장소의 실행 스크립트는 독립적으로 작성했습니다. 사용한 학습 모델 카드에는 Apache-2.0이 표시돼 있습니다. 상세 출처와 확인 범위는 [연구 기반 기록](docs/research-base.md)을 참고하세요.
