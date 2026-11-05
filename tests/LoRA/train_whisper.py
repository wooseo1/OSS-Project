"""Whisper 를 구음장애 음성으로 LoRA 또는 전체 파인튜닝.

입력: make_manifest.py 가 만든 train.jsonl / valid.jsonl
출력: <output-dir>/adapter (LoRA) 또는 <output-dir>/full (전체), metrics.json, train_config.json

예)
  python train_whisper.py --train-manifest manifests/train.jsonl --valid-manifest manifests/valid.jsonl ^
      --base-model openai/whisper-small --mode lora --output-dir outputs/lora-small
"""
import argparse
import inspect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

from audio_utils import audio_duration, load_audio
from metadata_manifest import build_manifests
from textutil import corpus_rates, normalize


class ManifestDataset(Dataset):
    """manifest(JSONL)를 읽어 필요할 때마다 오디오를 로드해 feature/label 로 변환"""

    def __init__(self, manifest, processor, max_seconds=30.0, train=False,
                 speed_aug=False, normalize_labels=False):
        self.processor = processor
        self.train = train
        self.speed_aug = speed_aug and train
        self.normalize_labels = normalize_labels
        rows, skipped = [], 0
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                dur = r.get("duration")
                if dur is None:
                    try:
                        dur = audio_duration(r["audio"])
                    except Exception:
                        skipped += 1
                        continue
                text = normalize(r["text"]) if normalize_labels else r["text"].strip()
                if not text or dur < 0.1 or dur > max_seconds:
                    skipped += 1
                    continue
                rows.append({"audio": r["audio"], "text": text})
        self.rows = rows
        print(f"[데이터] {manifest}: 사용 {len(rows)}개, 제외 {skipped}개 (길이 0.1~{max_seconds}초 범위 밖 등)")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        speed = random.choice([0.9, 1.0, 1.1]) if self.speed_aug else 1.0
        audio = load_audio(r["audio"], speed=speed)
        feats = self.processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
        labels = self.processor.tokenizer(r["text"]).input_ids[:448]
        return {"input_features": feats, "labels": labels}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # 모델이 labels 를 오른쪽으로 밀면서 decoder_start_token 을 붙이므로 중복 제거
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def make_compute_metrics(tokenizer):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = np.where(pred.label_ids == -100, tokenizer.pad_token_id, pred.label_ids)
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        cer, wer = corpus_rates(list(zip(label_str, pred_str)))
        return {"cer": cer, "wer": wer}

    return compute_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-manifest", help="기존 JSONL 학습 manifest")
    ap.add_argument("--valid-manifest", help="기존 JSONL 검증 manifest")
    ap.add_argument("--data-root", help="metadata.csv 와 전처리 WAV가 있는 데이터 폴더")
    ap.add_argument("--disease", choices=["larynx", "neuro", "speech", "all"],
                    help="--data-root 사용 시 학습할 데이터 그룹")
    ap.add_argument("--train-set", choices=["full", "user_10min", "user_20min", "user_30min"],
                    default="full", help="학습 데이터 양 (data-root 사용 시)")
    ap.add_argument("--test-set", choices=["full", "user_10min"], default="full",
                    help="생성할 별도 테스트 manifest")
    ap.add_argument("--base-model", default="openai/whisper-small",
                    help="HF 모델 id 또는 로컬 폴더 (HF 형식이어야 함, CT2 변환본은 불가)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", choices=["lora", "full"], default="lora")
    ap.add_argument("--language", default="ko")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--max-steps", type=int, default=-1, help="동작 확인용 (예: 5). -1 이면 epochs 사용")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None, help="기본: lora 5e-4, full 1e-5")
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--eval-steps", type=int, default=0, help="0 이면 epoch 마다 평가/저장")
    ap.add_argument("--max-eval-samples", type=int, default=0,
                    help="CPU smoke test용 검증 샘플 상한 (0 이면 전체)")
    ap.add_argument("--generation-max-length", type=int, default=128,
                    help="평가 시 생성 토큰 최대 길이")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", default="q_proj,v_proj", help="쉼표로 구분 (예: q_proj,k_proj,v_proj,out_proj)")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--speed-aug", action="store_true", help="학습 때 말속도 0.9/1.0/1.1 무작위 변형")
    ap.add_argument("--normalize-labels", action="store_true", help="정답 문장의 구두점 제거 후 학습")
    ap.add_argument("--grad-checkpointing", action="store_true", help="메모리 절약 (속도는 느려짐)")
    ap.add_argument("--bf16", action="store_true", help="fp16 대신 bf16 (Ampere 이상 GPU)")
    ap.add_argument("--no-eval-before", action="store_true", help="학습 전 baseline 평가 생략")
    ap.add_argument("--num-workers", type=int, default=0, help="Windows 는 0 권장")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.data_root:
        if args.train_manifest or args.valid_manifest:
            ap.error("--data-root 과 --train-manifest/--valid-manifest 는 함께 사용할 수 없습니다")
        if not args.disease:
            ap.error("--data-root 사용 시 --disease 를 지정하세요 (larynx/neuro/speech/all)")
        manifests = build_manifests(args.data_root, Path(args.output_dir) / "manifests", args.disease,
                        train_source=args.train_set, test_source=args.test_set)
        args.train_manifest = str(manifests["train"])
        args.valid_manifest = str(manifests["valid"])
    elif not args.train_manifest or not args.valid_manifest:
        ap.error("--data-root 또는 --train-manifest 와 --valid-manifest 를 모두 지정하세요")

    set_seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    print(f"[장치] {'CUDA ' + torch.cuda.get_device_name(0) if use_cuda else 'CPU (매우 느림, Colab 등 GPU 권장)'}")

    processor = WhisperProcessor.from_pretrained(args.base_model, language=args.language, task="transcribe")
    processor.tokenizer.set_prefix_tokens(language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.config.use_cache = False if args.grad_checkpointing else model.config.use_cache

    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model

        if args.grad_checkpointing:
            # 인코더 입력은 임베딩이 아니라서 gradient checkpointing 시 grad 가 끊김 -> 훅으로 보정
            def make_inputs_require_grad(module, inputs, output):
                output.requires_grad_(True)

            model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[t.strip() for t in args.lora_targets.split(",") if t.strip()],
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        lr = args.lr or 5e-4
    else:
        lr = args.lr or 1e-5

    train_ds = ManifestDataset(args.train_manifest, processor, args.max_seconds, train=True,
                               speed_aug=args.speed_aug, normalize_labels=args.normalize_labels)
    valid_ds = ManifestDataset(args.valid_manifest, processor, args.max_seconds, train=False,
                               normalize_labels=args.normalize_labels)
    if args.max_eval_samples > 0 and len(valid_ds) > args.max_eval_samples:
        indices = random.Random(args.seed).sample(range(len(valid_ds)), args.max_eval_samples)
        valid_ds.rows = [valid_ds.rows[index] for index in indices]
        print(f"[검증] smoke test 샘플 {len(valid_ds)}개 사용")
    if len(train_ds) == 0 or len(valid_ds) == 0:
        raise SystemExit("train 또는 valid 데이터가 비었음. manifest 를 확인하세요.")

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)

    # transformers 버전별 인자 이름 차이 대응
    ta_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
    strategy = "steps" if args.eval_steps > 0 else "epoch"
    ta_kwargs = dict(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        fp16=use_cuda and not args.bf16,
        bf16=use_cuda and args.bf16,
        gradient_checkpointing=args.grad_checkpointing,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        save_strategy=strategy,
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        remove_unused_columns=False,   # PEFT 모델은 forward 시그니처가 가려져서 필수
        label_names=["labels"],        # 위와 같은 이유로 필수
        dataloader_num_workers=args.num_workers,
        report_to="none",
        seed=args.seed,
    )
    ta_kwargs[eval_key] = strategy
    if args.eval_steps > 0:
        ta_kwargs["eval_steps"] = args.eval_steps
        ta_kwargs["save_steps"] = args.eval_steps
    training_args = Seq2SeqTrainingArguments(**ta_kwargs)

    tr_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    tok_kw = ({"processing_class": processor.feature_extractor} if "processing_class" in tr_params
              else {"tokenizer": processor.feature_extractor})
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor.tokenizer),
        **tok_kw,
    )

    metrics = {}
    if not args.no_eval_before:
        print("[학습 전] baseline 평가 중 ...")
        metrics["before"] = trainer.evaluate()
        print(f"[학습 전] CER {metrics['before'].get('eval_cer'):.4f}  WER {metrics['before'].get('eval_wer'):.4f}")

    trainer.train()
    metrics["after"] = trainer.evaluate()
    print(f"[학습 후] CER {metrics['after'].get('eval_cer'):.4f}  WER {metrics['after'].get('eval_wer'):.4f}")

    save_dir = out_dir / ("adapter" if args.mode == "lora" else "full")
    if args.mode == "lora":
        trainer.model.save_pretrained(str(save_dir))   # 어댑터 가중치만 저장 (수 MB~수십 MB)
    else:
        trainer.save_model(str(save_dir))
    processor.save_pretrained(str(save_dir))
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "lr_used": lr, "save_dir": str(save_dir)}, f, ensure_ascii=False, indent=2)
    print(f"[완료] 저장: {save_dir}")


if __name__ == "__main__":
    main()
