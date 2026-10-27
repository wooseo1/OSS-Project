# 파일 전체 평가 방법

## 평가 대상

AI Hub에서 받은 WAV/JSON 샘플 4쌍을 대상으로 기본 Whisper-small과 선행연구의 질환 통합 학습 모델을 비교한다. 모델의 출처와 리비전은 [연구 기반 기록](research-base.md)을 따른다. 새 모델을 학습하는 실험은 아니다.

짧은 구간의 정답은 없지만 파일 전체 Transcript가 있으므로 **전체 WAV를 입력하고 전체 Transcript와 비교**한다. 원본 전체에서 음성 구간을 탐지하더라도 정답을 일부 삭제하거나 추론 출력에 맞춰 정답을 바꾸지 않는다.

## 실행

저장소 루트에서 실행한다. 아래 예시는 앞선 README의 모델 준비가 끝난 상태를 가정한다.

```sh
.venv/bin/python -m pip install -r requirements-eval.txt
.venv/bin/python -m unittest discover -s tests -v

.venv/bin/python scripts/evaluate_full.py \
  --data-root ../datasets/aihub-dysarthria \
  --baseline ../models/whisper-small-ct2 \
  --adapted ../models/whisper-dysarthria-general-ct2 \
  --output-dir outputs/full-eval \
  --no-vad-first-file

.venv/bin/python scripts/evaluate_full.py \
  --data-root ../datasets/aihub-dysarthria \
  --baseline ../models/whisper-small-ct2 \
  --adapted ../models/whisper-dysarthria-general-ct2 \
  --output-dir outputs/utterance-eval \
  --segmented --no-vad-sample 2
```

이미 결과가 있는 출력 폴더에는 기록을 덮어쓰지 않고 중단한다. `results.json`에 진행 상태를 기록하고 파일별 완료 시 저장한다. `status=completed`인지 확인해야 전체 실험 완료로 간주한다. 중단된 실험은 새 폴더에서 재실행한다.

## 비교 조건

| 조건 | 처리 |
| --- | --- |
| `vad_0.3` | 전체 파일에서 VAD로 무음을 제거하고 faster-whisper의 기본 묶음 처리로 추론 |
| `vad_utterances` | 같은 VAD 구간을 각각 따로 추론하고 시간 순서대로 연결 |
| `no_vad` | 선택한 전체 파일을 VAD 없이 추론 |

VAD는 Silero를 사용하며 threshold 0.3, min_silence_duration_ms 500, speech_pad_ms 400으로 고정한다. 모든 모델에서 CPU int8, 4 threads, 한국어, beam 5, temperature 0, 이전 텍스트 조건 끔을 사용한다.

`vad_utterances`는 반복 문장을 포함한 샘플에서 누락이 발견된 후 추가한 **탐색적 처리 방식 비교**다. 별도 검증 세트로 선택한 최종 설정이 아니다. VAD가 누락한 음성은 두 VAD 조건 모두에서 누락 오류로 남는다. 무음 제거로 인식할 내용이 사라졌는지 확인하려고 선택한 파일을 `no_vad`에서도 실행한다.

## 지표 정의

계산 도구는 [JiWER 4.0.0](https://github.com/jitsi/jiwer)이다. 정답과 출력에 같은 정규화를 적용한다.

1. Unicode NFC 정규화, 영문 소문자화.
2. Unicode P 범주의 문장부호 제거, 연속 공백 정리.
3. CER: 공백을 제외한 글자 단위 편집 거리 / 정답 글자 수.
4. WER: 공백으로 나눈 어절 단위 편집 거리 / 정답 어절 수.

CER은 한글 음절 단위이며 자모 단위가 아니다. WER은 형태소 분석을 적용하지 않으므로 한국어 띄어쓰기 차이에도 민감하다. 숫자를 한글로 바꾸거나 철자·정답 오탈자를 자동 교정하지 않는다. 문장부호를 보존한 보조 지표도 별도로 저장한다.

`통합(micro)`은 파일별 대치·삭제·삽입 개수를 합산해 전체 정답 단위 수로 나눈다. `파일 평균(macro)`은 파일별 오류율의 단순 평균이다. 두 값을 혼용하지 않는다. 삽입이 많으면 오류율이 100%를 넘을 수 있다.

## 기록과 해석

`results.json`에는 입력 음성·라벨·모델·토크나이저·실행 코드의 SHA256, 환경·라이브러리 버전, 실행 조건, 전체 정답과 출력, 타임스탬프, 오류 개수와 집계 결과를 저장한다. SHA256은 같은 입력 파일인지 확인하는 용도다. 원본과 결과 파일은 로컬에 보관한다.

처리 시간은 음성 디코딩·VAD·추론을 포함하고 모델 로딩은 제외한다. 워밍업과 시스템 부하를 통제하지 않았으므로 정확한 속도 벤치마크로 해석하지 않는다.

샘플 수와 화자 수가 작고, 선행 모델의 학습 데이터와 중복 여부가 미확인이다. 낭독 음성의 결과를 실생활 대화 성능으로 확대하지 않는다. 질환 중증도도 검증되지 않았다. 논문 수치 재현, 독립 테스트 성능, 화자 개인화 효과를 주장하지 않는다.

## 다음 실험에 필요한 것

화자와 문장 중복을 통제한 학습/검증/평가 분할, 발화별 음성·정답의 사람 검수, 각 화자의 적응용 음성과 평가용 음성이 필요하다. 이를 확보한 뒤 LoRA·Adapter를 학습하고 동일 평가 데이터에서 개선폭과 비용을 측정한다.
