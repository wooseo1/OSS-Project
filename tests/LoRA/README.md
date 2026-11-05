# Whisper LoRA 학습 코드 (구음장애 음성)

전처리된 WAV + 라벨 JSON → manifest → LoRA/전체 파인튜닝 → 평가(오류 패턴 분석) → faster-whisper 변환

## 구성

| 파일 | 역할 |
|---|---|
| `make_manifest.py` | WAV와 JSON을 `File_id`로 짝지어 train/valid/test JSONL 생성 |
| `train_whisper.py` | LoRA(`--mode lora`) 또는 전체 파인튜닝(`--mode full`) |
| `evaluate.py` | 여러 모델을 같은 test로 평가 + 자모 단위 오류 패턴 CSV |
| `merge_convert.py` | LoRA 병합 → HF 저장 → faster-whisper(CT2) 변환 |
| `textutil.py`, `audio_utils.py` | CER/WER, 자모 분석, 오디오 로드 공통 모듈 |

아래 명령은 저장소 루트에서 실행하며, 학습 코드는 `tests/LoRA/`에 있습니다.

## 0. 설치 (Windows PowerShell)

```powershell
# PyTorch는 CUDA 버전에 맞춰 먼저 설치: https://pytorch.org/get-started/locally/
.venv\Scripts\python.exe -m pip install -r tests\LoRA\requirements-train.txt
```

- GPU 없으면 학습이 매우 느림. Colab(T4 이상)에서 같은 코드를 실행하는 것을 권장
- 프로젝트·데이터·모델 폴더는 OneDrive 밖(예: `C:\dev`, `C:\data`, `C:\models`)에 두기

## 1. 학습 실행

전처리된 `dataset` 폴더에 metadata CSV와 각 WAV가 있으면 trainer가 CSV를 읽어 manifest를 만들고 바로 LoRA 학습을 시작합니다. `--disease`는 `larynx`, `neuro`, `speech`, `all` 중 데이터 그룹에 맞게 지정하세요. 개인화 실험에서는 `--train-set user_10min`과 `--test-set user_10min`으로 사용자별 10분 학습·테스트셋을 선택할 수 있습니다.

```powershell
.venv\Scripts\python.exe tests\LoRA\train_whisper.py `
  --data-root C:\Users\kookm\Downloads\dataset\dataset --disease larynx `
  --base-model openai/whisper-small --mode lora `
  --output-dir C:\Users\kookm\Downloads\dataset\runs\lora-larynx `
  --epochs 5 --batch-size 4 --grad-accum 4
```

Intel/CPU 환경에서는 우선 아래처럼 사용자 10분 학습과 작은 검증으로 실행하세요. 생성된 test manifest는 별도 전후 비교용이며 학습에 사용하지 않습니다.

```powershell
.venv\Scripts\python.exe tests\LoRA\train_whisper.py `
  --data-root C:\Users\kookm\Downloads\dataset\dataset --disease larynx `
  --train-set user_10min --test-set user_10min `
  --base-model openai/whisper-tiny --mode lora `
  --output-dir C:\Users\kookm\Downloads\dataset\runs\lora-user10-tiny-50 `
  --max-steps 50 --eval-steps 25 --max-eval-samples 4 --generation-max-length 32 `
  --batch-size 1 --grad-accum 1 --warmup-steps 5
```

기본 Whisper와 LoRA 모델을 같은 사용자 test 샘플에서 비교합니다. 처음에는 `--max-samples 32`로 빠르게 점검하고, 전체 평가에서는 해당 옵션을 생략하세요.

```powershell
.venv\Scripts\python.exe tests\LoRA\evaluate.py `
  --manifest C:\Users\kookm\Downloads\dataset\runs\lora-user10-tiny-50\manifests\test.jsonl `
  --out-dir C:\Users\kookm\Downloads\dataset\runs\lora-user10-tiny-50\comparison `
  --hf before=openai/whisper-tiny `
  --adapter after=openai/whisper-tiny::C:\Users\kookm\Downloads\dataset\runs\lora-user10-tiny-50\adapter `
  --max-samples 8 --max-new-tokens 32 --batch-size 1
```

`summary.csv`에 두 모델의 CER/WER가 저장되고, 각 모델의 문장별 출력도 같은 결과 폴더에 저장됩니다. 전체 평가를 하기 전에는 모델 선택에 사용하지 않은 held-out test manifest인지 확인하세요.

전체 test 평가에서 `--max-samples`와 `--max-new-tokens`를 생략하면 됩니다. CPU 50-step 예비 실행 결과는 held-out 252개에서 base CER/WER 2.8501/1.6817, LoRA CER/WER 0.4848/0.6817이었습니다. 짧은 학습 실험 결과이며 일반화 성능으로 해석하지 마세요.

생성 manifest는 `<output-dir>\manifests`에 저장됩니다. 테스트 데이터 누수를 막으려면 test manifest는 학습·검증에 사용하지 마세요.

CUDA가 없는 CPU 환경에서는 학습이 매우 느립니다. PyTorch가 아직 정상 설치되지 않았다면 CUDA 버전에 맞는 PyTorch를 먼저 설치한 뒤 아래 학습 의존성을 설치하세요.

```powershell
.venv\Scripts\python.exe -m pip install -r tests\LoRA\requirements-train.txt
```

기존 JSONL manifest가 있다면 아래 방식도 그대로 사용할 수 있습니다.

## 2. JSONL manifest 만들기

라벨 JSON에서 **정답 문장이 들어있는 키 경로**를 모르는 상태라, 먼저 구조를 확인:

```powershell
.venv\Scripts\python.exe tests\LoRA\make_manifest.py --data-root C:\data\aihub-dysarthria --inspect
```

출력된 구조에서 정답 문장 키를 찾아 `--text-key`에 지정 (점으로 연결, 리스트는 숫자 인덱스. 예: `Transcript.0.text`)

```powershell
.venv\Scripts\python.exe tests\LoRA\make_manifest.py --data-root C:\data\aihub-dysarthria `
  --out-dir manifests --text-key <정답문장_키경로> --speaker-key <화자_키경로>
```

- `--speaker-key`를 주면 **화자 단위로 분할**(test에 처음 보는 화자가 들어감). 화자가 3명 미만이면 파일 단위로 분할되고 경고 출력
- 결과: `manifests/train.jsonl`, `valid.jsonl`, `test.jsonl`
- AI Hub 데이터와 manifest(음성 경로 포함)는 **GitHub에 올리지 않기** (`.gitignore`)

manifest 한 줄 형식 (직접 만들어도 됨):
```json
{"audio": "C:\\data\\a001.wav", "text": "약 먹을 시간이야", "speaker": "S01", "duration": 3.2}
```

## 3. JSONL manifest로 동작 확인 (5스텝)

```powershell
.venv\Scripts\python.exe tests\LoRA\train_whisper.py --train-manifest manifests\train.jsonl `
  --valid-manifest manifests\valid.jsonl --base-model openai/whisper-small --mode lora `
  --output-dir outputs\smoke --max-steps 5 --eval-steps 5 --batch-size 2 --grad-accum 1 --warmup-steps 1
```

## 4. JSONL manifest로 LoRA 학습

```powershell
.venv\Scripts\python.exe tests\LoRA\train_whisper.py --train-manifest manifests\train.jsonl `
  --valid-manifest manifests\valid.jsonl --base-model openai/whisper-small --mode lora `
  --output-dir outputs\lora-small --epochs 5 --batch-size 8 --grad-accum 2 --speed-aug
```

- 학습 전(baseline) / 학습 후 valid CER·WER가 `outputs\lora-small\metrics.json`에 저장됨
- 메모리 부족(CUDA OOM): `--batch-size 4 --grad-accum 4`, 그래도 부족하면 `--grad-checkpointing`
- 어댑터만 저장되므로 용량이 작음 (`outputs\lora-small\adapter`)
- 기존에 받은 `yoona-J/ASR_Whisper_Disease_General`(HF 형식 폴더)을 base로 써서 그 위에 LoRA를 얹는 것도 가능:
  `--base-model C:\models\whisper-dysarthria-general`

### 전체 파인튜닝과 비교 (제안서의 비교 실험)

```powershell
.venv\Scripts\python.exe tests\LoRA\train_whisper.py ... --mode full --output-dir outputs\full-small
```
전체 파인튜닝은 메모리를 훨씬 많이 쓰므로 `--batch-size 4 --grad-checkpointing` 권장. 학습률 기본값은 lora 5e-4, full 1e-5.

## 5. 평가 + 오류 패턴 분석

```powershell
.venv\Scripts\python.exe tests\LoRA\evaluate.py --manifest manifests\test.jsonl --out-dir outputs\eval `
  --hf baseline=openai/whisper-small `
  --adapter lora=openai/whisper-small::outputs\lora-small\adapter `
  --hf full=outputs\full-small\full
```

`outputs\eval`에 생기는 파일
- `summary.csv`: 모델별 CER/WER
- `by_speaker.csv`: 화자별 CER (중증도·화자 간 편차 확인)
- `samples_<모델>.csv`: 문장별 정답/인식 결과
- `jamo_errors_<모델>.csv`: **어떤 음소(초성/중성/종성)가 얼마나 틀리는지**
- `jamo_confusions_<모델>.csv`: **무엇이 무엇으로 바뀌는지** (예: `초ㄷ → 초ㅌ`)

엑셀에서 열면 한글이 깨지지 않게 `utf-8-sig`로 저장함.

## 6. 서버에 연결 (faster-whisper)

```powershell
.venv\Scripts\python.exe tests\LoRA\merge_convert.py --base-model openai/whisper-small `
  --adapter outputs\lora-small\adapter --out-hf outputs\lora-small\merged --ct2-out outputs\lora-small\ct2
$env:MODEL_DIR = "C:\...\outputs\lora-small\ct2"
.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

`--base-model`은 `outputs\lora-small\train_config.json`의 `base_model` 값과 같아야 함.

## 주의

- `--base-model`은 **HF 형식**(config.json + safetensors)이어야 함. faster-whisper CT2 변환본(`*-ct2`)은 학습 불가
- 학습 데이터 길이는 30초 이하만 사용(Whisper 한계). 초과분은 자동 제외하고 개수를 출력
- 기본 학습률·LoRA 설정(r=32, alpha=64, q_proj/v_proj)은 널리 쓰는 값이지, 이 데이터에 맞춰 튜닝한 값이 아님. valid CER을 보며 `--lr`, `--lora-r`, `--lora-targets`, `--epochs` 조정
- 한국어 인식기라서 라벨 문장 표기 규칙(숫자 표기, 띄어쓰기)이 일관돼야 CER이 의미 있음. 필요하면 `--normalize-labels`
