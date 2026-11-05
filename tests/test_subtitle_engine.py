import math
import unittest

from subtitle_engine import SubtitleCue, SubtitleStyle, make_ass


class SubtitleEngineTests(unittest.TestCase):
    def test_style_is_scaled_to_frame_and_text_cannot_inject_tags(self):
        result = make_ass([SubtitleCue(1, 2.5, 'မင်္ဂလာပါ {\\pos(0,0)}')], 1920, 1080, SubtitleStyle(25, 80, 5), duration=3)
        self.assertIn('PlayResX: 1920', result)
        self.assertIn('Noto Sans Myanmar,54.00', result)
        self.assertIn(r'{\pos(480,864)}', result)
        self.assertNotIn(r'{\pos(0,0)}', result)

    def test_rejects_invalid_cues_and_style(self):
        with self.assertRaises(ValueError):
            make_ass([SubtitleCue(2, 1, 'စာ')], 100, 100, SubtitleStyle())
        with self.assertRaises(ValueError):
            make_ass([SubtitleCue(0, 2, 'စာ')], 100, 100, SubtitleStyle(), duration=1)
        with self.assertRaises(ValueError):
            make_ass([SubtitleCue(0, 1, 'စာ')], 100, 100, SubtitleStyle(math.nan, 20, 4))

    def test_time_rounding_carries_to_next_minute(self):
        result = make_ass([SubtitleCue(59.999, 61, 'စာ')], 100, 100, SubtitleStyle())
        self.assertIn('0:01:00.00,0:01:01.00', result)
