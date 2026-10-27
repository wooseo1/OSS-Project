import unittest
from unittest.mock import patch
from dataclasses import dataclass
import numpy as np

from scripts.evaluate_full import aggregate, error_counts, normalize, transcribe_complete


class EvaluationTests(unittest.TestCase):
    def test_korean_unicode_and_punctuation(self):
        self.assertEqual(normalize("가,  나!\n"), "가 나")
        self.assertEqual(error_counts("가 나!", "가나")["cer"]["rate"], 0)
        self.assertGreater(error_counts("가 나!", "가나")["wer"]["rate"], 0)

    def test_substitution_deletion_and_insertion(self):
        self.assertEqual(error_counts("가나다", "가라다")["cer"]["substitutions"], 1)
        self.assertEqual(error_counts("가나다", "가다")["cer"]["deletions"], 1)
        self.assertEqual(error_counts("가나다", "가나다라")["cer"]["insertions"], 1)

    def test_missing_speech_counts_as_deletions(self):
        result = error_counts("가나다", "")
        self.assertEqual(result["cer"]["rate"], 1)
        self.assertEqual(result["cer"]["deletions"], 3)

    def test_insertions_may_exceed_100_percent(self):
        self.assertGreater(error_counts("가", "가나다라")["cer"]["rate"], 1)

    def test_corpus_weighting_is_not_file_average(self):
        rows = [{"metrics": error_counts("가", "나")},
                {"metrics": error_counts("가나다라마바사아자차", "가나다라마바사아자차")}]
        result = aggregate(rows)["cer"]
        self.assertAlmostEqual(result["micro_rate"], 1/11)
        self.assertAlmostEqual(result["macro_rate"], .5)

    def test_punctuation_only_reference_rejected(self):
        with self.assertRaises(ValueError):
            error_counts("...", "가")

    def test_independent_spans_preserve_repetitions_and_absolute_time(self):
        @dataclass
        class Segment:
            start: float
            end: float
            text: str

        class FakeModel:
            def transcribe(self, audio, **kwargs):
                return iter([Segment(0, .5, "문 열어")]), None

        with patch("faster_whisper.audio.decode_audio", return_value=np.zeros(48000,dtype=np.float32)), \
             patch("faster_whisper.vad.get_speech_timestamps", return_value=[
                 {"start": 0, "end": 8000}, {"start": 32000, "end": 40000}]):
            segments, info = transcribe_complete(FakeModel(), "unused.wav", {}, True, True)
            items = list(segments)
        self.assertEqual([s.text for s in items], ["문 열어", "문 열어"])
        self.assertEqual([s.start for s in items], [0, 2])
        self.assertEqual(info.duration, 3)
        self.assertEqual(info.duration_after_vad, 1)


if __name__ == "__main__":
    unittest.main()
