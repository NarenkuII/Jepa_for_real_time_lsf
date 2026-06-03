from __future__ import annotations

from pathlib import Path

import numpy as np

FACE_SIGN_RELEVANT = (
    1,
    4,
    6,
    10,
    13,
    14,
    17,
    33,
    37,
    39,
    40,
    46,
    52,
    55,
    61,
    70,
    80,
    81,
    82,
    87,
    88,
    91,
    95,
    105,
    107,
    133,
    145,
    159,
    161,
    263,
    269,
    270,
    276,
    291,
    308,
    311,
    312,
    317,
    318,
    324,
)


def _landmarks_to_array(landmarks, count: int, indices: tuple[int, ...] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if indices is None:
        indices = tuple(range(count))
    arr = np.zeros((len(indices), 6), dtype=np.float32)
    conf = np.zeros((len(indices),), dtype=np.float32)
    valid = np.zeros((len(indices),), dtype=bool)
    if landmarks is None:
        return arr, conf, valid
    for out_i, lm_i in enumerate(indices):
        if lm_i >= len(landmarks.landmark):
            continue
        lm = landmarks.landmark[lm_i]
        visibility = float(getattr(lm, "visibility", 1.0))
        presence = float(getattr(lm, "presence", visibility))
        c = min(visibility, presence)
        arr[out_i] = [lm.x, lm.y, lm.z, c, visibility, 1.0]
        conf[out_i] = c
        valid[out_i] = c > 0
    return arr, conf, valid


def extract_video_keypoints(video: str, start: float | None = None, end: float | None = None, config: dict | None = None) -> dict:
    try:
        import cv2
        import mediapipe as mp
    except Exception as exc:
        raise RuntimeError("MediaPipe extraction requires optional dependencies: pip install -e '.[vision]'") from exc

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    first_frame = int((start or 0.0) * fps)
    last_frame = int(end * fps) if end is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

    face_subset = (config or {}).get("keypoints", {}).get("face_subset", "sign_relevant")
    face_indices = FACE_SIGN_RELEVANT if face_subset == "sign_relevant" else tuple(range(468))
    use_pose = (config or {}).get("keypoints", {}).get("use_pose", True)
    use_face = (config or {}).get("keypoints", {}).get("use_face", True)
    use_hands = (config or {}).get("keypoints", {}).get("use_hands", True)

    frames, confidences, masks = [], [], []
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1, refine_face_landmarks=True)
    frame_idx = first_frame
    try:
        while True:
            if last_frame and frame_idx >= last_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)
            parts = []
            conf_parts = []
            mask_parts = []
            if use_pose:
                a, c, m = _landmarks_to_array(result.pose_landmarks, 33)
                parts.append(a)
                conf_parts.append(c)
                mask_parts.append(m)
            if use_face:
                a, c, m = _landmarks_to_array(result.face_landmarks, len(face_indices), face_indices)
                parts.append(a)
                conf_parts.append(c)
                mask_parts.append(m)
            if use_hands:
                for hand in (result.left_hand_landmarks, result.right_hand_landmarks):
                    a, c, m = _landmarks_to_array(hand, 21)
                    parts.append(a)
                    conf_parts.append(c)
                    mask_parts.append(m)
            frames.append(np.concatenate(parts, axis=0))
            confidences.append(np.concatenate(conf_parts, axis=0))
            masks.append(np.concatenate(mask_parts, axis=0))
            frame_idx += 1
    finally:
        holistic.close()
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from {video}")
    return {
        "keypoints": np.stack(frames).astype(np.float32),
        "confidence": np.stack(confidences).astype(np.float32),
        "valid_mask": np.stack(masks).astype(bool),
        "fps": np.float32(fps),
        "topology_name": "mediapipe_holistic",
        "source_video": video,
        "start": np.float32(start or 0.0),
        "end": np.float32((end if end is not None else frame_idx / fps)),
    }


def save_keypoints_npz(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
