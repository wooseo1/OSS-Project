import unittest
from types import SimpleNamespace

from scripts.meet_caption_server import join_segment_text


class MeetCaptionServerTests(unittest.TestCase):
    def test_join_segment_text_strips_and_omits_empty_segments(self):
        segments = [SimpleNamespace(text=" 안녕하세요 "), SimpleNamespace(text="  "),
                    SimpleNamespace(text="반갑습니다")]
        self.assertEqual(join_segment_text(segments), "안녕하세요 반갑습니다")

    def test_join_segment_text_returns_empty_for_silence(self):
        self.assertEqual(join_segment_text([SimpleNamespace(text=" ")]), "")


if __name__ == "__main__":
    unittest.main()