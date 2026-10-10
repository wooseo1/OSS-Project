# 재현 대상과 사용 조건

2026-10-11 확인.

- 기존 논문 관리대장의 재현 1순위: Diagnosis-aware multitask fine-tuning of Whisper for dysarthric speech recognition.
- 연구 코드: https://github.com/yoona-J/Diagnosis-Aware_Multitask_Fine-Tuning_of_Whisper_for_DSR
- 확인한 코드 커밋: `4624249a217e85ff91752c2f689eca4ede15686a`
- 코드 저장소에는 명시적 LICENSE 파일이 없다. 별도 로컬 클론을 참고하며 이 저장소에 원본 노트북을 복사하지 않았다.
- 실행 모델: https://huggingface.co/yoona-J/ASR_Whisper_Disease_General
- 모델 리비전: `304fe2952840b5b96ca3d4399a43d024912d4096`
- 모델 카드의 라이선스 표기: Apache-2.0. 모델 파일과 카드를 로컬에 보관한다.
- 기본 모델 실행 파일: https://huggingface.co/Systran/faster-whisper-small
- 기본 모델 리비전: `536b0662742c02347bc0e980a01041f333bce120`

저자의 전처리·추론 노트북은 발화 단위 WAV/TXT와 별도 분석 CSV를 기대한다.
현재 받은 샘플은 장시간 WAV와 파일 전체 Transcript JSON이므로 그대로 실행하면 대응하지 않는다.
독립 실행 스크립트에서 파일 ID로 짝을 확인하고, 실제 WAV 헤더를 기준으로 16kHz로 재표본화한다.
JSON에는 48kHz, 실제 WAV에는 44.1kHz로 기록된 불일치가 확인됐다.

현재 단계는 기본 모델과 질환 통합 학습 모델의 같은 구간 추론이다.
연구의 질환 분류기/질환별 라우팅/가중 CER를 재현한 결과가 아니다.
원 학습 데이터와 현재 샘플의 중복 여부가 확인되지 않아 독립 평가로 해석할 수 없다.
정량 평가에는 발화별 음성·정답 정렬, 화자와 문장 중복을 고려한 학습/평가 분리가 필요하다.
AI Hub 원본·환자 메타데이터·추론 결과는 공개 GitHub에 포함하지 않는다.

## 로컬 실행 확인

2026-10-11: WAV/JSON 4쌍을 검증하고 각 파일의 30~60초를 두 모델에서 실행했다.
총 8개의 추론 출력, 각 클립의 16kHz/480,000개 샘플, 정량 지표 미계산 상태를 확인했다.
CPU int8, 한국어 고정, beam 5, temperature 0, VAD 끔, 이전 텍스트 조건 끔을 동일하게 적용했다.
`pip check`로 의존성 충돌이 없음을 확인했다.
CTranslate2 4.8.2 변환기는 Transformers 4.54.1과 dtype 인자 충돌이 있어 4.57.6으로 고정했다.
이 환경은 논문 원본 환경과 다르므로 수치 재현을 주장하지 않는다.
