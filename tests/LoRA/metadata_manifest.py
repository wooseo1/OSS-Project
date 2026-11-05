"""Convert the preprocessed dataset metadata CSVs to Whisper JSONL manifests."""
import csv
import json
from pathlib import Path


TRAIN_SOURCES = {
    "full": "01_train",
    "user_10min": "04_user/train_10min",
    "user_20min": "04_user/train_20min",
    "user_30min": "04_user/train_30min",
}
TEST_SOURCES = {
    "full": "03_test",
    "user_10min": "04_user/test_10min",
}


def build_manifests(data_root, output_dir, disease, train_source="full", test_source="full"):
    root = Path(data_root).expanduser().resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if train_source not in TRAIN_SOURCES:
        raise ValueError(f"지원하지 않는 train source: {train_source}")
    if test_source not in TEST_SOURCES:
        raise ValueError(f"지원하지 않는 test source: {test_source}")
    split_directories = {
        "train": TRAIN_SOURCES[train_source],
        "valid": "02_val",
        "test": TEST_SOURCES[test_source],
    }
    manifests = {}

    for split_name, directory in split_directories.items():
        metadata_path = root / directory / "metadata.csv"
        if not metadata_path.is_file():
            if split_name == "test":
                continue
            raise FileNotFoundError(f"메타데이터 CSV가 없습니다: {metadata_path}")

        rows = []
        with metadata_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required = {"file_name", "transcription", "disease"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{metadata_path} 필수 컬럼 누락: {', '.join(sorted(missing))}")

            for line_number, record in enumerate(reader, start=2):
                if disease != "all" and record["disease"].strip().casefold() != disease:
                    continue
                text = record["transcription"].strip()
                if not text:
                    raise ValueError(f"{metadata_path}:{line_number} 정답 문장이 비어 있습니다")

                audio_path = (root / directory / record["file_name"]).resolve()
                try:
                    audio_path.relative_to(root)
                except ValueError as exc:
                    raise ValueError(f"데이터 폴더 밖의 오디오 경로: {record['file_name']}") from exc
                if not audio_path.is_file():
                    raise FileNotFoundError(f"오디오 파일이 없습니다: {audio_path}")

                item = {"audio": str(audio_path), "text": text}
                speaker = record.get("speaker", "").strip()
                if speaker:
                    item["speaker"] = speaker
                try:
                    item["duration"] = float(record["duration"])
                except (KeyError, TypeError, ValueError):
                    pass
                rows.append(item)

        if not rows and split_name != "test":
            raise ValueError(f"{metadata_path}에서 '{disease}' 학습 항목을 찾지 못했습니다")
        if rows:
            manifest_path = output / f"{split_name}.jsonl"
            with manifest_path.open("w", encoding="utf-8", newline="\n") as destination:
                for item in rows:
                    destination.write(json.dumps(item, ensure_ascii=False) + "\n")
            manifests[split_name] = manifest_path
            print(f"[{split_name}] {len(rows)}개 -> {manifest_path}")

    return manifests