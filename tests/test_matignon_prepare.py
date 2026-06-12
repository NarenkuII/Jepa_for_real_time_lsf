from pathlib import Path

from tools.prepare_matignon_lsf import parse_vtt, split_name, timestamp_seconds


def test_matignon_subtitle_helpers_are_deterministic():
    text = """WEBVTT

00:00:01.000 --> 00:00:03.500
Bonjour <i>tout le monde</i>
"""
    cues = parse_vtt(text)

    assert cues == [{"start": 1.0, "end": 3.5, "text_fr": "Bonjour tout le monde"}]
    assert timestamp_seconds("01:02:03,500") == 3723.5
    assert split_name("video-a") == split_name("video-a")
