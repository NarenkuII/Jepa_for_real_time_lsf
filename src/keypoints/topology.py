from __future__ import annotations

from dataclasses import dataclass

SIGN_RELEVANT_FACE_LANDMARKS = tuple(
    sorted(
        set(
            # Mouth/lips.
            (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415)
            # Eyes.
            + (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
            + (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
            # Eyebrows.
            + (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)
            + (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)
            # Nose bridge/tip.
            + (1, 2, 4, 5, 6, 197, 195, 98, 327)
        )
    )
)


@dataclass(frozen=True)
class JointGroup:
    name: str
    indices: tuple[int, ...]


@dataclass(frozen=True)
class KeypointTopology:
    name: str
    num_joints: int
    groups: dict[str, JointGroup]
    edges: tuple[tuple[int, int], ...]
    landmarks: dict[str, int]


def mediapipe_holistic_topology(face_subset: str = "sign_relevant") -> KeypointTopology:
    pose = tuple(range(0, 33))
    face_count = len(SIGN_RELEVANT_FACE_LANDMARKS) if face_subset == "sign_relevant" else 468
    face = tuple(range(33, 33 + face_count))
    left_hand = tuple(range(33 + len(face), 33 + len(face) + 21))
    right_hand = tuple(range(33 + len(face) + 21, 33 + len(face) + 42))
    groups = {
        "pose": JointGroup("pose", pose),
        "face": JointGroup("face", face),
        "left_hand": JointGroup("left_hand", left_hand),
        "right_hand": JointGroup("right_hand", right_hand),
        "torso": JointGroup("torso", (11, 12, 23, 24)),
        "arms": JointGroup("arms", (11, 12, 13, 14, 15, 16)),
    }
    hand_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
    edges = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
    edges += [(left_hand[a], left_hand[b]) for a, b in hand_edges]
    edges += [(right_hand[a], right_hand[b]) for a, b in hand_edges]
    landmarks = {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_hip": 23,
        "right_hip": 24,
        "nose": 0,
        "left_wrist": left_hand[0],
        "right_wrist": right_hand[0],
        "left_index_mcp": left_hand[5],
        "right_index_mcp": right_hand[5],
    }
    return KeypointTopology("mediapipe_holistic", 33 + len(face) + 42, groups, tuple(edges), landmarks)


def synthetic_topology() -> KeypointTopology:
    return mediapipe_holistic_topology("sign_relevant")
