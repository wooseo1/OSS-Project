"""WAV + 라벨 JSON 을 File_id 로 짝지어 학습용 manifest(JSONL) 생성.

한 줄 형식: {"audio": "절대경로.wav", "text": "정답 문장", "speaker": "화자(선택)", "duration": 초(선택)}

사용 순서
 1) JSON 구조 확인:   python make_manifest.py --data-root DATA --inspect
 2) 정답 문장이 들어있는 키 경로를 --text-key 로 지정해서 manifest 생성
    (키 경로는 점(.)으로 연결, 리스트는 숫자 인덱스. 예: Transcript.0.text)
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def get_by_path(obj, path):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur[part]
        else:
            raise KeyError(path)
    return cur


def describe(obj, indent=0, max_depth=4):
    """JSON 구조(키, 타입, 짧은 미리보기)를 출력"""
    pad = "  " * indent
    if indent > max_depth:
        print(f"{pad}...")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                print(f"{pad}{k}: {type(v).__name__}")
                describe(v, indent + 1, max_depth)
            else:
                preview = str(v)
                preview = preview[:40] + ("..." if len(preview) > 40 else "")
                print(f"{pad}{k}: {type(v).__name__} = {preview!r}")
    elif isinstance(obj, list):
        print(f"{pad}[list 길이 {len(obj)}]")
        if obj:
            describe(obj[0], indent + 1, max_depth)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def split_items(items, ratios, seed, warn):
    """화자 정보가 충분하면 화자 단위로, 아니면 파일 단위로 train/valid/test 분할"""
    rng = random.Random(seed)
    speakers = {it.get("speaker") for it in items if it.get("speaker")}
    all_have_speaker = all(it.get("speaker") for it in items)
    if all_have_speaker and len(speakers) >= 3:
        groups = defaultdict(list)
        for it in items:
            groups[it["speaker"]].append(it)
        keys = list(groups)
        rng.shuffle(keys)
        total = len(items)
        targets = [total * r for r in ratios]
        buckets = [[], [], []]
        for idx, k in enumerate(keys):
            remaining = len(keys) - idx
            empties = [i for i in range(3) if not buckets[i]]
            if len(empties) >= remaining:
                target = empties[0]   # 남은 화자 수가 비어있는 분할 수 이하면 빈 분할부터 채움 (valid/test 가 비지 않도록)
            else:
                deficits = [targets[i] - len(buckets[i]) for i in range(3)]
                target = deficits.index(max(deficits))
            buckets[target].extend(groups[k])
        return buckets, "speaker"
    warn("화자 정보가 없거나 화자가 3명 미만이라 파일 단위로 분할합니다 (같은 화자가 train/test 에 섞일 수 있음)")
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_valid = max(1, round(n * ratios[1])) if n >= 3 else 0
    n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
    n_train = n - n_valid - n_test
    return [items[:n_train], items[n_train:n_train + n_valid], items[n_train + n_valid:]], "file"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="WAV 와 JSON 이 들어있는 상위 폴더")
    ap.add_argument("--out-dir", default="manifests", help="manifest 출력 폴더")
    ap.add_argument("--id-key", default="File_id", help="JSON 안의 파일 ID 키 경로 (기본 File_id)")
    ap.add_argument("--text-key", help="JSON 안의 정답 문장 키 경로")
    ap.add_argument("--speaker-key", default=None, help="JSON 안의 화자 키 경로 (선택, 화자 단위 분할용)")
    ap.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VALID", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict", action="store_true", help="짝이 안 맞는 파일이 있으면 중단")
    ap.add_argument("--inspect", action="store_true", help="첫 JSON 구조만 출력하고 종료")
    args = ap.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        sys.exit(f"폴더가 없음: {root}")

    wavs, dup_wav = {}, 0
    json_files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".wav":
            if p.stem in wavs:
                dup_wav += 1
            wavs[p.stem] = p
        elif suf == ".json":
            json_files.append(p)
    json_files.sort()
    print(f"WAV {len(wavs)}개, JSON {len(json_files)}개 발견 (WAV 파일명 중복 {dup_wav})")
    if not json_files:
        sys.exit("JSON 이 없음")

    if args.inspect:
        print(f"\n첫 JSON: {json_files[0].name}")
        describe(read_json(json_files[0]))
        return
    if not args.text_key:
        sys.exit("--text-key 가 필요함. 먼저 --inspect 로 JSON 구조를 확인하세요.")

    try:
        import soundfile as sf
    except ImportError:
        sf = None

    items, problems = [], defaultdict(int)
    used = set()
    for jp in json_files:
        try:
            data = read_json(jp)
            file_id = str(get_by_path(data, args.id_key))
            text = get_by_path(data, args.text_key)
        except Exception as e:  # 키 경로 오류, JSON 파싱 오류 등
            problems["JSON 읽기/키 경로 오류"] += 1
            if args.strict:
                sys.exit(f"{jp}: {e!r}")
            continue
        if not isinstance(text, str) or not text.strip():
            problems["정답 문장이 문자열이 아니거나 비어 있음"] += 1
            continue
        stem = Path(file_id).stem
        wav = wavs.get(stem)
        if wav is None:
            problems["JSON 에 대응하는 WAV 없음"] += 1
            if args.strict:
                sys.exit(f"WAV 없음: {file_id}")
            continue
        if stem in used:
            problems["같은 File_id 가 여러 JSON 에 등장"] += 1
            if args.strict:
                sys.exit(f"중복 File_id: {file_id}")
            continue
        used.add(stem)
        item = {"audio": str(wav.resolve()), "text": text.strip()}
        if args.speaker_key:
            try:
                item["speaker"] = str(get_by_path(data, args.speaker_key))
            except Exception:
                pass
        if sf is not None:
            try:
                item["duration"] = round(float(sf.info(str(wav)).duration), 3)
            except Exception:
                problems["WAV 헤더 읽기 실패"] += 1
        items.append(item)

    unpaired_wav = len(wavs) - len(used)
    if unpaired_wav:
        problems["짝 없는 WAV"] += unpaired_wav
        if args.strict:
            sys.exit(f"짝 없는 WAV {unpaired_wav}개")
    for k, v in problems.items():
        print(f"[경고] {k}: {v}")
    if not items:
        sys.exit("만들어진 항목이 0개. --id-key / --text-key 를 확인하세요.")

    (train, valid, test), mode = split_items(items, args.split, args.seed, lambda m: print(f"[경고] {m}"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = sum(i.get("duration", 0) for i in items) / 3600
    print(f"\n총 {len(items)}개 (약 {hours:.2f}시간), 분할 방식: {mode}")
    print(f"train {len(train)} / valid {len(valid)} / test {len(test)}  ->  {out.resolve()}")
    if not valid or not test:
        print("[경고] valid 또는 test 가 비었음. 데이터가 너무 적음 (샘플 4쌍이면 동작 확인용으로만 사용)")


if __name__ == "__main__":
    main()
