# 협업과 커밋 안내

한 가지 자막 기능 목적의 변경을 한 커밋으로 기록합니다. 데이터·녹음·모델 파일은 저장소에 추가하지 않습니다.

## 작업 시작

```sh
git status --short
git switch main
git pull --ff-only origin main
git switch -c feat/meet-caption-change
```

## 변경 확인과 커밋

```sh
git status --short
git diff
git add meet-captions scripts/meet_caption_server.py tests/test_meet_caption_server.py requirements-meet.txt
git diff --cached --check
git diff --cached
git commit -m "feat: Meet 자막 연결 개선"
```

커밋 메시지는 한국어로 작성합니다. `git add`에는 이번 변경 파일만 지정하고, 커밋 전에 파일 목록을 확인합니다.

## 실행 확인

Python 브리지 변경은 아래 테스트를 실행합니다.

```sh
python -m unittest discover -s tests -v
```

확장 변경은 `meet-captions/manifest.json`이 올바른 JSON인지 확인하고 Chrome에서 압축해제된 확장 프로그램으로 로드해 점검합니다.

## 공유

```sh
git push -u origin feat/meet-caption-change
```

브랜치에서 PR을 열고 변경 내용과 확인 결과를 적습니다. 공유된 브랜치의 기록을 강제로 덮어쓰지 않습니다.
