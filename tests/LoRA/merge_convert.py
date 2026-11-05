"""LoRA 어댑터 병합 -> HF 모델 저장 -> (선택) faster-whisper(CTranslate2) 변환.

변환된 폴더를 서버의 MODEL_DIR 에 지정하면 크롬 확장 서버에서 바로 사용 가능.

예)
  python merge_convert.py --base-model openai/whisper-small --adapter outputs/lora-small/adapter ^
      --out-hf outputs/lora-small/merged --ct2-out outputs/lora-small/ct2
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", required=True, help="학습 때 쓴 base 모델 (train_config.json 의 base_model)")
    ap.add_argument("--adapter", required=True, help="학습 결과 adapter 폴더")
    ap.add_argument("--out-hf", required=True, help="병합된 HF 모델 저장 폴더")
    ap.add_argument("--ct2-out", default=None, help="지정하면 CT2 변환까지 수행")
    ap.add_argument("--quantization", default="float16", help="CT2 양자화 (float16 / int8 / int8_float16 ...)")
    ap.add_argument("--force", action="store_true", help="CT2 출력 폴더가 이미 있어도 덮어씀")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print("[1/3] 기본 모델 + 어댑터 로드")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(model, args.adapter)
    print("[2/3] 병합 (merge_and_unload)")
    model = model.merge_and_unload()

    out_hf = Path(args.out_hf)
    out_hf.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_hf))
    try:
        processor = WhisperProcessor.from_pretrained(args.adapter)
    except Exception:
        processor = WhisperProcessor.from_pretrained(args.base_model)
    processor.save_pretrained(str(out_hf))
    print(f"      저장: {out_hf.resolve()}")

    if not args.ct2_out:
        print("[3/3] CT2 변환 생략 (--ct2-out 미지정)")
        return

    print("[3/3] faster-whisper(CTranslate2) 변환")
    try:
        from ctranslate2.converters import TransformersConverter

        TransformersConverter(
            str(out_hf), copy_files=["tokenizer.json", "preprocessor_config.json"]
        ).convert(args.ct2_out, quantization=args.quantization, force=args.force)
        print(f"      저장: {Path(args.ct2_out).resolve()}")
    except Exception as e:
        print(f"[실패] 파이썬 API 변환 오류: {e!r}")
        print("      아래 명령으로 직접 변환 가능:")
        print(f"      ct2-transformers-converter --model {out_hf} --output_dir {args.ct2_out} "
              f"--copy_files tokenizer.json preprocessor_config.json --quantization {args.quantization}")


if __name__ == "__main__":
    main()
