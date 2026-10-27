"""Convert a downloaded research checkpoint for local CPU inference."""
import argparse
from pathlib import Path
from ctranslate2.converters import TransformersConverter
from transformers import WhisperTokenizerFast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = str(args.source.resolve())
    tokenizer = WhisperTokenizerFast.from_pretrained(source, local_files_only=True)
    # Generate the tokenizer before conversion so the output is fully offline.
    tokenizer.save_pretrained(source)
    TransformersConverter(source, copy_files=["tokenizer.json", "preprocessor_config.json"]).convert(
        str(args.output.resolve()), quantization="int8")


if __name__ == "__main__":
    main()
