"""여러 모델을 같은 test manifest 로 평가하고, 오류 패턴(어떤 음소가 어떻게 틀리는지)까지 분석.

모델 지정 방식 (여러 번 지정 가능, 이름=경로 형태)
  --hf    NAME=PATH            HF 형식 Whisper (허브 id 또는 로컬 폴더, 전체 파인튜닝/병합 모델 포함)
  --adapter NAME=BASE::ADAPTER  기본 모델 + LoRA 어댑터 (병합 없이 평가)
  --ct2   NAME=PATH            faster-whisper(CTranslate2) 변환 모델

예)
  python evaluate.py --manifest manifests/test.jsonl --out-dir outputs/eval ^
      --hf baseline=openai/whisper-small ^
      --adapter lora=openai/whisper-small::outputs/lora-small/adapter

출력(out-dir)
  summary.csv                    모델별 CER/WER
  by_speaker.csv                 모델별·화자별 CER (manifest 에 speaker 가 있을 때)
  samples_<NAME>.csv             문장별 정답/인식결과/CER
  jamo_errors_<NAME>.csv         자모 토큰별 오류율 (어떤 음소가 잘 틀리는지)
  jamo_confusions_<NAME>.csv     자모 대체 쌍 (무엇이 무엇으로 바뀌는지)
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from audio_utils import load_audio
from textutil import TokenErrorStats, cer_counts, corpus_rates


def parse_specs(items, with_base=False):
    out = []
    for s in items or []:
        if "=" not in s:
            raise SystemExit(f"형식 오류 (NAME=값 이어야 함): {s}")
        name, val = s.split("=", 1)
        if with_base:
            if "::" not in val:
                raise SystemExit(f"--adapter 형식은 NAME=BASE::ADAPTER_DIR : {s}")
            base, adapter = val.split("::", 1)
            out.append((name, base, adapter))
        else:
            out.append((name, val))
    return out


def load_rows(manifest):
    rows = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def transcribe_hf(model, processor, rows, args):
    import torch

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    outs = []
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        audios = [load_audio(r["audio"]) for r in batch]
        feats = processor.feature_extractor(audios, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(device=device, dtype=dtype)
        with torch.no_grad():
            ids = model.generate(input_features=feats, language=args.language, task="transcribe",
                                 max_new_tokens=args.max_new_tokens, num_beams=args.beams)
        outs.extend(processor.batch_decode(ids, skip_special_tokens=True))
        print(f"  {min(i + args.batch_size, len(rows))}/{len(rows)}", end="\r")
    print()
    return [o.strip() for o in outs]


def hf_device_dtype():
    import torch

    return ("cuda", torch.float16) if torch.cuda.is_available() else ("cpu", torch.float32)


def run_hf(path, rows, args):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device, dtype = hf_device_dtype()
    processor = WhisperProcessor.from_pretrained(path)
    model = WhisperForConditionalGeneration.from_pretrained(path, torch_dtype=dtype).to(device).eval()
    return transcribe_hf(model, processor, rows, args)


def run_adapter(base, adapter, rows, args):
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device, dtype = hf_device_dtype()
    try:
        processor = WhisperProcessor.from_pretrained(adapter)
    except Exception:
        processor = WhisperProcessor.from_pretrained(base)
    model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter).to(device).eval()
    return transcribe_hf(model, processor, rows, args)


def run_ct2(path, rows, args):
    from faster_whisper import WhisperModel

    model = WhisperModel(path, device=args.device, compute_type=args.compute_type)
    outs = []
    for i, r in enumerate(rows, 1):
        segs, _ = model.transcribe(load_audio(r["audio"]), language=args.language,
                                   beam_size=args.beams, vad_filter=False,
                                   condition_on_previous_text=False)
        outs.append(" ".join(s.text.strip() for s in segs).strip())
        print(f"  {i}/{len(rows)}", end="\r")
    print()
    return outs


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: 엑셀에서 한글 깨짐 방지
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="평가할 manifest (보통 test.jsonl)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hf", action="append", help="NAME=PATH")
    ap.add_argument("--adapter", action="append", help="NAME=BASE::ADAPTER_DIR")
    ap.add_argument("--ct2", action="append", help="NAME=PATH")
    ap.add_argument("--language", default="ko")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="빠른 비교용 샘플 상한 (0 이면 manifest 전체)")
    ap.add_argument("--device", default="auto", help="ct2 용 (auto/cpu/cuda)")
    ap.add_argument("--compute-type", default="default", help="ct2 용 (default/int8/float16 ...)")
    ap.add_argument("--min-count", type=int, default=5, help="콘솔 요약에서 자모 오류율을 볼 최소 등장 횟수")
    args = ap.parse_args()

    rows = load_rows(args.manifest)
    if args.max_samples > 0:
        rows = rows[:args.max_samples]
    if not rows:
        raise SystemExit("manifest 가 비었음")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for name, path in parse_specs(args.hf):
        jobs.append((name, lambda p=path: run_hf(p, rows, args)))
    for name, base, adapter in parse_specs(args.adapter, with_base=True):
        jobs.append((name, lambda b=base, a=adapter: run_adapter(b, a, rows, args)))
    for name, path in parse_specs(args.ct2):
        jobs.append((name, lambda p=path: run_ct2(p, rows, args)))
    if not jobs:
        raise SystemExit("평가할 모델이 없음 (--hf / --adapter / --ct2)")

    summary, by_speaker = [], []
    for name, fn in jobs:
        print(f"\n=== {name} ===")
        t0 = time.time()
        hyps = fn()
        elapsed = time.time() - t0
        pairs = [(r["text"], h) for r, h in zip(rows, hyps)]
        cer, wer = corpus_rates(pairs)
        summary.append({"model": name, "n": len(rows), "CER": round(cer, 4), "WER": round(wer, 4),
                        "seconds": round(elapsed, 1)})
        print(f"CER {cer:.4f}  WER {wer:.4f}  ({elapsed:.1f}s)")

        stats = TokenErrorStats()
        sample_rows = []
        spk_acc = defaultdict(lambda: [0, 0])
        for r, h in zip(rows, hyps):
            e, n = cer_counts(r["text"], h)
            stats.add(r["text"], h)
            sample_rows.append({"audio": Path(r["audio"]).name, "speaker": r.get("speaker", ""),
                                "reference": r["text"], "hypothesis": h,
                                "cer": round(e / n, 4) if n else 0.0})
            if r.get("speaker"):
                spk_acc[r["speaker"]][0] += e
                spk_acc[r["speaker"]][1] += n
        write_csv(out / f"samples_{name}.csv", sample_rows,
                  ["audio", "speaker", "reference", "hypothesis", "cer"])
        write_csv(out / f"jamo_errors_{name}.csv", stats.rows(),
                  ["token", "count", "errors", "error_rate", "substitutions", "deletions", "top_confusions"])
        write_csv(out / f"jamo_confusions_{name}.csv", stats.pair_rows(), ["ref", "hyp", "count"])
        for spk, (e, n) in sorted(spk_acc.items()):
            by_speaker.append({"model": name, "speaker": spk, "n_chars": n, "CER": round(e / n, 4) if n else 0.0})

        worst = [r for r in stats.rows() if r["count"] >= args.min_count][:8]
        if worst:
            print("자모 오류율 상위 (등장 %d회 이상):" % args.min_count)
            for r in worst:
                print(f"  {r['token']:<4} {r['error_rate']:.2f}  ({r['errors']}/{r['count']})  -> {r['top_confusions']}")

    write_csv(out / "summary.csv", summary, ["model", "n", "CER", "WER", "seconds"])
    if by_speaker:
        write_csv(out / "by_speaker.csv", by_speaker, ["model", "speaker", "n_chars", "CER"])
    print("\n=== 요약 ===")
    for s in summary:
        print(f"{s['model']:<16} CER {s['CER']:.4f}  WER {s['WER']:.4f}  n={s['n']}")
    print(f"결과 폴더: {out.resolve()}")


if __name__ == "__main__":
    main()
