import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from metadata_manifest import build_manifests


class MetadataManifestTests(unittest.TestCase):
    def test_builds_audio_text_manifests_for_selected_disease(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            output = Path(temporary) / "manifests"
            for directory in ("01_train", "02_val", "03_test"):
                split = root / directory
                split.mkdir(parents=True)
                rows = [
                    {"file_name": "larynx/sample.wav", "transcription": "정답 문장", "disease": "larynx", "speaker": "S01", "duration": "1.25"},
                    {"file_name": "neuro/other.wav", "transcription": "다른 그룹", "disease": "neuro", "speaker": "S02", "duration": "1.50"},
                ]
                for row in rows:
                    audio = split / row["file_name"]
                    audio.parent.mkdir(parents=True, exist_ok=True)
                    audio.touch()
                with (split / "metadata.csv").open("w", encoding="utf-8", newline="") as metadata:
                    writer = csv.DictWriter(metadata, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

            with patch("builtins.print"):
                manifests = build_manifests(root, output, "larynx")

            self.assertEqual(set(manifests), {"train", "valid", "test"})
            for manifest in manifests.values():
                rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["text"], "정답 문장")
                self.assertEqual(rows[0]["speaker"], "S01")
                self.assertTrue(Path(rows[0]["audio"]).is_file())
                self.assertEqual(rows[0]["duration"], 1.25)

    def test_selects_user_train_and_user_test_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            for directory in ("04_user/train_10min", "02_val", "04_user/test_10min"):
                split = root / directory
                split.mkdir(parents=True)
                row = {"file_name": "larynx/sample.wav", "transcription": "정답", "disease": "larynx"}
                audio = split / row["file_name"]
                audio.parent.mkdir(parents=True)
                audio.touch()
                with (split / "metadata.csv").open("w", encoding="utf-8", newline="") as metadata:
                    writer = csv.DictWriter(metadata, fieldnames=row.keys())
                    writer.writeheader()
                    writer.writerow(row)

            with patch("builtins.print"):
                manifests = build_manifests(root, Path(temporary) / "manifests", "larynx",
                                            train_source="user_10min", test_source="user_10min")

            self.assertIn("test", manifests)
            self.assertTrue(Path(json.loads(manifests["train"].read_text(encoding="utf-8"))["audio"]).is_file())
            self.assertTrue(Path(json.loads(manifests["test"].read_text(encoding="utf-8"))["audio"]).is_file())


if __name__ == "__main__":
    unittest.main()