"""Inspect AI Hub WAV/JSON pairs and compare local CTranslate2 models.

This is an independently written runner. Upstream research notebooks stay in
their separate reference clone because no explicit code license was found.
"""
import argparse
import json
from pathlib import Path
import time
import unicodedata
import wave


def normalized_name(value):
    return unicodedata.normalize("NFC", Path(value).name).casefold()


def inspect_pairs(root):
    audio = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() == ".wav":
            key = normalized_name(path.name)
            if key in audio:
                raise ValueError(f"Duplicate audio filename: {path.name}")
            audio[key] = path
    records = []
    used = set()
    for label in sorted(root.rglob("*.json")):
        data = json.loads(label.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or "Transcript" not in data:
            continue
        key = normalized_name(data.get("File_id", label.with_suffix(".wav").name))
        if key not in audio or key in used:
            raise ValueError(f"Missing or duplicate audio pair: {label.name}")
        text = data["Transcript"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty or invalid transcript: {label.name}")
        path = audio[key]
        with wave.open(str(path)) as wav:
            rate = wav.getframerate()
            seconds = wav.getnframes() / rate
            channels = wav.getnchannels()
        declared = data.get("Meta_info", {}).get("SamplingRate")
        records.append({
            "sample": f"sample_{len(records)+1:02d}",
            "audio": str(path.resolve()), "label": str(label.resolve()),
            "duration_seconds": seconds, "sample_rate": rate,
            "channels": channels, "declared_sample_rate": declared,
            "transcript_characters": len(text),
            "reference_scope": "whole_file",
        })
        used.add(key)
    if not records or used != set(audio):
        raise ValueError("No valid pairs, or unpaired audio files remain")
    return records


def read_clip(path, seconds, start=0):
    """Decode only the requested prefix, using actual headers and resampling."""
    import av
    import numpy as np
    limit = round((start + seconds) * 16000)
    chunks, count = [], 0
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                values = converted.to_ndarray().reshape(-1)
                values = values[:limit-count]
                chunks.append(values)
                count += len(values)
            if count >= limit:
                break
        if count < limit:
            for converted in resampler.resample(None):
                values = converted.to_ndarray().reshape(-1)[:limit-count]
                chunks.append(values)
                count += len(values)
    if count <= round(start * 16000):
        raise ValueError(f"Empty decoded clip: {path}")
    return np.concatenate(chunks)[round(start * 16000):].astype(np.float32) / 32768.0


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--adapted", type=Path)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--start-seconds", type=float, default=30)
    args = parser.parse_args()
    if not 0 < args.seconds <= 30:
        parser.error("Smoke test clips must be between 0 and 30 seconds")
    if args.start_seconds < 0:
        parser.error("--start-seconds must be nonnegative")
    if bool(args.baseline) != bool(args.adapted):
        parser.error("Provide both --baseline and --adapted, or neither")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = inspect_pairs(args.data_root)
    write_json(args.output_dir/"inventory.json", records)
    print(f"Validated {len(records)} WAV/JSON pairs", flush=True)
    if not args.baseline:
        return
    for model_path in [args.baseline, args.adapted]:
        if not (model_path/"model.bin").is_file():
            raise ValueError(f"Expected local CTranslate2 model directory: {model_path}")
    from faster_whisper import WhisperModel
    results = {
        "purpose": "smoke_test_not_paper_reproduction",
        "metric_status": "not_scored_no_segment_aligned_reference",
        "cer": None, "wer": None,
        "conditions": {"language": "ko", "beam_size": 5, "vad_filter": False,
                       "condition_on_previous_text": False, "temperature": 0,
                       "compute_type": "int8", "device": "cpu"},
        "models": {"baseline": str(args.baseline.resolve()),
                   "adapted": str(args.adapted.resolve())}, "predictions": [],
    }
    for name, model_path in [("baseline", args.baseline), ("adapted", args.adapted)]:
        model = WhisperModel(str(model_path.resolve()), device="cpu", compute_type="int8",
                             cpu_threads=4, local_files_only=True)
        for record in records:
            clip = read_clip(record["audio"], args.seconds, args.start_seconds)
            started = time.perf_counter()
            segments, _ = model.transcribe(clip, language="ko", task="transcribe",
                                          beam_size=5, vad_filter=False, temperature=0,
                                          condition_on_previous_text=False)
            segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
            result = {"sample": record["sample"], "model": name,
                      "clip_start_seconds": args.start_seconds, "clip_seconds": len(clip)/16000,
                      "inference_seconds": time.perf_counter()-started,
                      "text": " ".join(s["text"].strip() for s in segments),
                      "segments": segments}
            results["predictions"].append(result)
            write_json(args.output_dir/"comparison.json", results)
            print(f"{name} {record['sample']}: {result['text']}", flush=True)
        del model
    lines = ["# AI Hub 샘플 추론 비교", "",
             f"각 파일의 {args.start_seconds:g}~{args.start_seconds+args.seconds:g}초 구간을 동일한 CPU int8 조건으로 추론했습니다.",
             "정답 텍스트는 파일 전체에 해당하므로 CER/WER를 계산하지 않았습니다.",
             "논문 성능 재현이나 개인화 학습 결과가 아닙니다. 학습 데이터 중복 여부도 미확인입니다.", ""]
    for record in records:
        lines.extend([f"## {record['sample']}", ""])
        for result in results["predictions"]:
            if result["sample"] == record["sample"]:
                lines.extend([f"**{result['model']}** ({result['inference_seconds']:.1f}초)",
                              "", result["text"] or "(출력 없음)", ""])
    (args.output_dir/"comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
