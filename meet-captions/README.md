# Whisper Google Meet Captions

로컬 마이크 음성을 기존 faster-whisper CTranslate2 모델로 인식해 Meet 화면에 자막으로 표시하는 Chrome 확장입니다. 자막 오버레이는 내 브라우저에서만 보입니다. 다른 참가자와 공유하려면 **채팅 입력창에 넣기**를 누른 뒤 문장을 확인하고 Meet에서 직접 전송하세요.

음성은 `127.0.0.1`에서 처리하고 파일로 저장하지 않습니다. Meet 참가자에게 채팅을 보내기 전에는 반드시 인식 내용을 확인하세요.

## 설치

프로젝트 루트에서 추가 의존성을 설치합니다.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-meet.txt
```

Chrome에서 `chrome://extensions`를 열고 개발자 모드를 켠 다음 **압축해제된 확장 프로그램을 로드합니다**를 선택해 이 `meet-captions` 폴더를 지정합니다. Google Meet 탭을 새로고침하세요.

## 실행

먼저 사용 가능한 마이크 장치 번호를 확인합니다.

```powershell
.venv\Scripts\python.exe scripts\meet_caption_server.py --list-devices
```

서버를 실행할 때 프로젝트에서 사용하는 로컬 faster-whisper 모델 폴더를 지정합니다.

```powershell
.venv\Scripts\python.exe scripts\meet_caption_server.py `
  --model C:\models\whisper-dysarthria-general-ct2 --input-device 1
```

`--input-device`는 선택 사항이며, 생략하면 Windows 기본 마이크를 사용합니다. 기본 청크 길이는 4초입니다. 서버를 종료하려면 `Ctrl+C`를 누릅니다. 확장 오버레이에 연결 상태와 인식 결과가 표시됩니다.

**채팅 입력창에 넣기**는 최근 한 문장만 Meet 채팅 입력창에 넣습니다. 자동 전송하지 않으며, 입력창을 찾지 못하거나 이미 작성 중인 내용이 있으면 변경하지 않습니다.

## 참고

- 모델 경로는 로컬 CTranslate2/faster-whisper 폴더여야 합니다. 서버는 네트워크에서 모델을 내려받지 않습니다.
- 로컬 자막을 Meet의 기본 자막 기능에 직접 주입하지는 않습니다. 오버레이는 본인 화면에만 나타납니다.
- 입력 지연과 인식 정확도는 모델 및 컴퓨터 성능에 따라 달라집니다. 마이크가 Meet과 다른 장치를 사용하면 `--list-devices`로 장치를 확인하세요.
- Meet의 UI와 접근성 라벨이 바뀌면 채팅 입력창 자동 탐색이 동작하지 않을 수 있습니다. 이 경우 **최근 문장 복사**를 사용하세요.