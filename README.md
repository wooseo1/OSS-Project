# Google Meet Whisper 자막 확장

로컬 faster-whisper 모델의 마이크 인식 결과를 Google Meet 화면에 자막으로 띄우는 Chrome 확장과 로컬 브리지입니다. 이 저장소는 자막 기능만 관리하며 모델 학습·변환·평가 코드는 포함하지 않습니다.

설치와 실행 방법은 [meet-captions 안내](meet-captions/README.md)를 참고하세요. 음성은 로컬에서 처리하며 파일로 저장하지 않습니다. 오버레이는 내 브라우저에만 표시되고, 다른 참가자에게 공유할 때는 인식 내용을 확인한 뒤 Meet 채팅에서 직접 전송해야 합니다.

## 구성

| 경로 | 역할 |
| --- | --- |
| `meet-captions/` | Chrome 확장과 사용 안내 |
| `scripts/meet_caption_server.py` | 로컬 마이크와 faster-whisper를 연결하는 브리지 |
| `requirements.txt` | Whisper 추론 의존성 |
| `requirements-meet.txt` | 자막 브리지 추가 의존성 |
| `tests/test_meet_caption_server.py` | 인식 문장 조립 테스트 |
| `CONTRIBUTING.md` | 협업과 커밋 안내 |
