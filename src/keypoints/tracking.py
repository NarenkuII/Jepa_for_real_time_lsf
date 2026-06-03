from __future__ import annotations


def detect_possible_left_right_flip(left_presence: float, right_presence: float, threshold: float = 0.2) -> bool:
    return abs(left_presence - right_presence) > threshold

