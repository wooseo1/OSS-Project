# 협업과 커밋 안내

한 가지 목적의 변경을 한 커밋으로 기록합니다. 코드와 실행 안내는 공유하고, 데이터·환자 메타데이터·모델·실험 원문 출력은 로컬에 보관합니다.

## 작업 시작

먼저 `git status`로 현재 변경을 확인하세요. 아래 예시는 미저장 변경이 없을 때 사용합니다. 변경이 있다면 현재 작업 브랜치에서 커밋하거나 Codex에 정리를 요청합니다.

```sh
git status
git switch main
git pull --ff-only origin main
git switch -c feat/utterance-manifest
```

브랜치 이름은 실제 작업에 맞게 정합니다. 예: `feat/utterance-manifest`, `fix/audio-pairing`, `docs/team-setup`.

## 변경 저장

```sh
git diff
git status --short
git add scripts/sample_compare.py
git diff --cached
git diff --cached --check
git commit -m "fix: 음성 파일과 정답의 연결 오류 수정"
```

`git add`에는 이번 작업 파일을 명시합니다. 새 파일은 `git diff`에 보이지 않으므로 파일 내용도 직접 확인합니다. 커밋 직전에는 포함된 파일과 내용을 확인합니다.

| 유형 | 예시 |
| --- | --- |
| `feat` | `feat: 발화별 평가 목록 생성 추가` |
| `fix` | `fix: 실제 음성 주파수로 재표본화` |
| `docs` | `docs: 팀원 실행 안내 정리` |
| `test` | `test: 잘못 연결된 라벨 검증 추가` |
| `chore` | `chore: 실행 의존성 버전 고정` |

기능과 실행에 필요한 설정은 함께 커밋해도 됩니다. 별개 기능과 대규모 문서 정리는 나눕니다. 고장 난 상태를 기본 브랜치에 합치지 않습니다.

작성자가 설정되지 않았다면 본인의 GitHub 이름과 이메일을 이 저장소에 설정합니다.

```sh
git config user.name "본인 GitHub 이름"
git config user.email "본인의 GitHub 이메일 또는 noreply 이메일"
```

## 팀에 공유

```sh
git push -u origin feat/utterance-manifest
```

GitHub에서 이 브랜치의 PR을 열고 변경 목적·실행 확인·남은 제한을 적습니다. 팀원이 검토한 뒤 `main`에 합칩니다. 리뷰 수정은 같은 브랜치에 새 커밋으로 추가합니다.

공유한 브랜치는 `push --force`, `reset --hard`로 기록을 덮어쓰지 않습니다. 충돌이 생기면 현재 변경을 보존하고 원인을 확인합니다.

## 실행 확인

설명만 바꿨다면 경로·명령·링크를 점검합니다. 데이터 처리를 바꿨다면 실제 WAV/JSON 짝 점검을 실행합니다. 모델 추론을 바꿨다면 같은 구간으로 비교 실행을 하고 조건을 기록합니다. 학습 데이터로 평가 수치를 보고하지 않습니다.

## Codex에 맡기기

- “현재 변경 확인하고 기능별로 커밋해 줘.”
- “팀원이 실행할 수 있게 점검하고 브랜치 올려서 PR 만들어 줘.”
- “최근 커밋에서 뭐가 바뀌었는지 설명해 줘.”

커밋은 내 컴퓨터에 변경 이력을 저장하고, 푸시는 GitHub에 공유합니다. 작업 보고에서 변경 내용, 확인 결과, 커밋 ID와 공유 여부를 함께 확인합니다.
