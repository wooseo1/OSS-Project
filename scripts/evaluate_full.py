"""Evaluate complete WAV files against complete AI Hub transcripts locally."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import unicodedata
from types import SimpleNamespace

from jiwer import process_characters, process_words

try:
    from .sample_compare import inspect_pairs, write_json
except ImportError:
    from sample_compare import inspect_pairs, write_json


def normalize(text, punctuation=False):
    text = unicodedata.normalize("NFC", text).lower()
    if not punctuation:
        text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return " ".join(text.split())


def error_counts(reference, prediction, punctuation=False):
    ref, hyp = normalize(reference, punctuation), normalize(prediction, punctuation)
    if not ref:
        raise ValueError("Reference must contain text after normalization")
    result = {}
    for name, values in [("cer", process_characters(ref.replace(" ", ""), hyp.replace(" ", ""))),
                         ("wer", process_words(ref, hyp))]:
        denominator = values.hits + values.substitutions + values.deletions
        errors = values.substitutions + values.deletions + values.insertions
        result[name] = {"rate": errors/denominator, "reference_units": denominator,
                        "substitutions": values.substitutions, "deletions": values.deletions,
                        "insertions": values.insertions, "hits": values.hits}
    return result


def aggregate(rows):
    result = {}
    for metric in ["cer", "wer"]:
        counts = {key: sum(row["metrics"][metric][key] for row in rows)
                  for key in ["reference_units", "substitutions", "deletions", "insertions", "hits"]}
        counts["micro_rate"] = sum(counts[k] for k in ["substitutions", "deletions", "insertions"])/counts["reference_units"]
        counts["macro_rate"] = sum(row["metrics"][metric]["rate"] for row in rows)/len(rows)
        result[metric] = counts
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(result):
    groups = defaultdict(list)
    for row in result["rows"]:
        groups[(row["condition"], row["model"])].append(row)
    return [{"condition": condition, "model": model, "files": len(rows),
             "audio_seconds": sum(r["audio_seconds"] for r in rows),
             "pipeline_seconds": sum(r["pipeline_seconds"] for r in rows),
             "metrics": aggregate(rows)} for (condition, model), rows in sorted(groups.items())]


def transcribe_complete(model, path, options, vad, segmented=False):
    common = dict(language="ko", task="transcribe", beam_size=5, temperature=0,
                  condition_on_previous_text=False)
    if not segmented or not vad:
        return model.transcribe(path, **common, vad_filter=vad, vad_parameters=options if vad else None)
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    audio = decode_audio(path, sampling_rate=16000)
    spans = get_speech_timestamps(audio, VadOptions(**options))
    info = SimpleNamespace(duration=len(audio)/16000,
                           duration_after_vad=sum(s["end"]-s["start"] for s in spans)/16000)

    def segments():
        for span in spans:
            chunk = audio[span["start"]:span["end"]]
            items, _ = model.transcribe(chunk, **common, vad_filter=False)
            offset = span["start"]/16000
            for item in items:
                yield replace(item, start=item.start+offset, end=item.end+offset)
    return segments(), info


def report(result):
    lines = ["# 파일 전체 음성인식 실험", "",
             "파일 전체 음성과 파일 전체 정답을 비교한 탐색 실험입니다. 짧은 구간의 정답을 추정하지 않았습니다.",
             "같은 화자로 보이는 4개 샘플이며 학습 데이터 중복 여부는 미확인입니다. 독립 테스트나 논문 수치 재현이 아닙니다.",
             "CER은 공백·문장부호 제외, WER은 문장부호 제외 후 공백 기준 어절 단위입니다. 낮을수록 좋습니다.", "",
             "| 조건 | 모델 | 파일 수 | 통합 CER | 통합 WER | 처리 시간(초) |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for group in result["summary"]:
        lines.append(f"| {group['condition']} | {group['model']} | {group['files']} | "
                     f"{group['metrics']['cer']['micro_rate']*100:.2f}% | "
                     f"{group['metrics']['wer']['micro_rate']*100:.2f}% | {group['pipeline_seconds']:.1f} |")
    lines.extend(["", "## 파일별 결과", "", "| 조건 | 샘플 | 모델 | CER | WER | VAD 유지 음성(초) |",
                  "| --- | --- | --- | ---: | ---: | ---: |"])
    for row in result["rows"]:
        lines.append(f"| {row['condition']} | {row['sample']} | {row['model']} | "
                     f"{row['metrics']['cer']['rate']*100:.2f}% | {row['metrics']['wer']['rate']*100:.2f}% | "
                     f"{row['duration_after_vad']:.1f} |")
    lines.extend(["", "## 조건과 해석", "",
                  "- 모든 파일을 잘라내지 않고 입력했습니다. VAD가 제거한 발화도 정답에서 제거하지 않아 누락 오류에 반영됩니다.",
                  "- 모델: 동일 Whisper-small 크기, CPU int8, 한국어, beam 5, temperature 0, 이전 텍스트 조건 끔.",
                  "- VAD: threshold 0.3, min_silence_duration_ms 500, speech_pad_ms 400. VAD 끔은 선택한 파일의 민감도 확인입니다.",
                  "- vad_utterances는 같은 VAD 구간을 발화별로 독립 인식하고 시간 순서대로 합칩니다. 정답을 이용해 구간을 선택하지 않습니다.",
                  "- 통합 지표는 파일별 오류 개수 합 / 정답 단위 개수 합입니다. 단순 평균과 구분합니다.",
                  "- 처리 시간은 음성 디코딩·VAD·추론을 포함하고 모델 로딩을 제외합니다. 통제된 속도 벤치마크는 아닙니다.",
                  "- 문장부호를 보존한 보조 지표, 대치·삭제·삽입 개수와 전체 출력은 results.json에 기록했습니다.",
                  "- 정답 자체의 오탈자와 띄어쓰기가 결과에 영향을 줄 수 있으며 사람 검수를 완료하지 않았습니다.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-vad-first-file", action="store_true")
    parser.add_argument("--no-vad-sample", type=int, default=0, help="1-based sample index; 0 disables")
    parser.add_argument("--segmented", action="store_true", help="Decode each VAD span independently")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a new empty output directory to preserve experiment records")
    paths = {"baseline": args.baseline.resolve(), "adapted": args.adapted.resolve()}
    for path in paths.values():
        if not (path/"model.bin").is_file() or not (path/"tokenizer.json").is_file():
            parser.error(f"Missing local model.bin or tokenizer.json: {path}")
    records = inspect_pairs(args.data_root)
    if not 0 <= args.no_vad_sample <= len(records):
        parser.error("--no-vad-sample is outside the sample list")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        record["audio_sha256"] = sha256(record["audio"])
        record["label_sha256"] = sha256(record["label"])
    result = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "exploratory_full_file_evaluation_not_independent_test",
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "packages": {p: importlib.metadata.version(p) for p in
                                           ["jiwer", "faster-whisper", "ctranslate2", "av", "numpy"]}},
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "evaluator_sha256": sha256(__file__),
              "conditions": {"language": "ko", "task": "transcribe", "beam_size": 5,
                             "temperature": 0, "condition_on_previous_text": False,
                             "device": "cpu", "compute_type": "int8", "cpu_threads": 4,
                             "independent_vad_spans": args.segmented,
                             "vad_parameters": {"threshold": 0.3, "min_silence_duration_ms": 500,
                                                "speech_pad_ms": 400}},
              "normalization": "NFC; lowercase; remove Unicode P punctuation; collapse whitespace; CER excludes spaces; WER uses whitespace words",
              "models": {name: {"path": str(path), "model_sha256": sha256(path/"model.bin"),
                                "tokenizer_sha256": sha256(path/"tokenizer.json")} for name,path in paths.items()},
              "inventory": records, "rows": [], "summary": []}
    write_json(args.output_dir/"results.json", result)
    from faster_whisper import WhisperModel
    for name, path in paths.items():
        model = WhisperModel(str(path), device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True)
        condition_name = "vad_utterances" if args.segmented else "vad_0.3"
        jobs = [(condition_name, record, True) for record in records]
        no_vad_samples = {1} if args.no_vad_first_file else set()
        if args.no_vad_sample:
            no_vad_samples.add(args.no_vad_sample)
        jobs.extend(("no_vad", records[i-1], False) for i in sorted(no_vad_samples))
        for condition, record, vad in jobs:
            print(f"START {name} {condition} {record['sample']} ({record['duration_seconds']:.1f}s)", flush=True)
            reference = json.loads(Path(record["label"]).read_text(encoding="utf-8-sig"))["Transcript"]
            started = last_log = time.perf_counter()
            segments, info = transcribe_complete(model, record["audio"], result["conditions"]["vad_parameters"],
                                                 vad, args.segmented)
            output = []
            for segment in segments:
                output.append({"start": segment.start, "end": segment.end, "text": segment.text,
                               "avg_logprob": segment.avg_logprob, "no_speech_prob": segment.no_speech_prob})
                now = time.perf_counter()
                if now-last_log >= 20:
                    print(f"PROGRESS {name} {condition} {record['sample']} t={segment.end:.1f}s elapsed={now-started:.1f}s", flush=True)
                    last_log = now
            elapsed = time.perf_counter()-started
            prediction = " ".join(s["text"].strip() for s in output)
            row = {"sample": record["sample"], "model": name, "condition": condition,
                   "audio_seconds": info.duration, "duration_after_vad": info.duration_after_vad,
                   "pipeline_seconds": elapsed, "reference": reference, "prediction": prediction,
                   "segments": output, "metrics": error_counts(reference,prediction),
                   "metrics_with_punctuation": error_counts(reference,prediction,True)}
            result["rows"].append(row)
            result["summary"] = summarize(result)
            write_json(args.output_dir/"results.json", result)
            (args.output_dir/"report.md").write_text(report(result), encoding="utf-8")
            print(f"DONE {name} {condition} {record['sample']} CER={row['metrics']['cer']['rate']:.4f} WER={row['metrics']['wer']['rate']:.4f} elapsed={elapsed:.1f}s", flush=True)
        del model
    result["status"] = "completed"
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output_dir/"results.json", result)
    (args.output_dir/"report.md").write_text(report(result), encoding="utf-8")


if __name__ == "__main__":
    main()
