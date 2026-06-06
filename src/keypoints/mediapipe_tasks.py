from __future__ import annotations

import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.keypoints.topology import SIGN_RELEVANT_FACE_LANDMARKS, KeypointTopology, mediapipe_holistic_topology


MODEL_URLS = {
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}


def _download_if_missing(path: Path, url: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def _safe_float(value, default: float = 1.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _landmarks_to_array(landmarks, indices: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros((len(indices), 6), dtype=np.float32)
    confidence = np.zeros((len(indices),), dtype=np.float32)
    valid = np.zeros((len(indices),), dtype=bool)
    if not landmarks:
        return values, confidence, valid
    for out_i, landmark_i in enumerate(indices):
        if landmark_i >= len(landmarks):
            continue
        landmark = landmarks[landmark_i]
        visibility = _safe_float(getattr(landmark, "visibility", 1.0), 1.0)
        presence = _safe_float(getattr(landmark, "presence", visibility), visibility)
        score = min(visibility, presence)
        values[out_i] = [landmark.x, landmark.y, landmark.z, score, visibility, 1.0]
        confidence[out_i] = score
        valid[out_i] = score > 0.0
    return values, confidence, valid


class MediaPipeTasksExtractor:
    """Pose + face subset + two-hand extractor using the MediaPipe Tasks API."""

    def __init__(self, model_dir: str | Path = "checkpoints/mediapipe", delegate: str = "gpu"):
        try:
            import mediapipe as mp
        except Exception as exc:
            raise RuntimeError("Install MediaPipe first: pip install mediapipe opencv-python") from exc

        self.mp = mp
        self.delegate_name = delegate
        model_dir = Path(model_dir)
        paths = {
            "pose": model_dir / "pose_landmarker_full.task",
            "hand": model_dir / "hand_landmarker.task",
            "face": model_dir / "face_landmarker.task",
        }
        for name, path in paths.items():
            _download_if_missing(path, MODEL_URLS[name])

        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions
        delegate_enum = base.Delegate.GPU if delegate == "gpu" else base.Delegate.CPU
        running_mode = vision.RunningMode.VIDEO
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=base(model_asset_path=str(paths["pose"]), delegate=delegate_enum),
                running_mode=running_mode,
                num_poses=1,
            )
        )
        self.hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base(model_asset_path=str(paths["hand"]), delegate=delegate_enum),
                running_mode=running_mode,
                num_hands=2,
            )
        )
        self.face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base(model_asset_path=str(paths["face"]), delegate=delegate_enum),
                running_mode=running_mode,
                num_faces=1,
            )
        )
        self.topology = mediapipe_holistic_topology("sign_relevant")
        self.last_timestamp_ms = -1

    @classmethod
    def create_with_fallback(cls, model_dir: str | Path, delegate: str = "gpu") -> "MediaPipeTasksExtractor":
        try:
            return cls(model_dir=model_dir, delegate=delegate)
        except Exception:
            if delegate != "gpu":
                raise
            return cls(model_dir=model_dir, delegate="cpu")

    def _process_frame(self, rgb: np.ndarray, timestamp_ms: int, mirrored_source: bool) -> dict:
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        pose_result = self.pose.detect_for_video(image, timestamp_ms)
        hand_result = self.hand.detect_for_video(image, timestamp_ms)
        face_result = self.face.detect_for_video(image, timestamp_ms)

        pose = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
        face = face_result.face_landmarks[0] if face_result.face_landmarks else None
        detected_hands: dict[str, object] = {}
        for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            label = handedness[0].category_name.lower() if handedness else ""
            # Tasks handedness assumes selfie-mirrored input. Correct labels for ordinary camera video.
            if not mirrored_source:
                label = "right" if label == "left" else "left" if label == "right" else label
            if label in ("left", "right"):
                detected_hands[label] = landmarks

        arrays = []
        confidences = []
        masks = []
        for landmarks, indices in (
            (pose, tuple(range(33))),
            (face, SIGN_RELEVANT_FACE_LANDMARKS),
            (detected_hands.get("left"), tuple(range(21))),
            (detected_hands.get("right"), tuple(range(21))),
        ):
            values, confidence, valid = _landmarks_to_array(landmarks, indices)
            arrays.append(values)
            confidences.append(confidence)
            masks.append(valid)

        keypoints = np.concatenate(arrays, axis=0)
        if mirrored_source:
            keypoints[:, 0] = 1.0 - keypoints[:, 0]
            keypoints[:, 2] *= -1.0
        return {
            "keypoints": keypoints,
            "confidence": np.concatenate(confidences),
            "valid_mask": np.concatenate(masks),
        }

    def extract_video(self, video: str | Path, mirrored_source: bool = False, fps_target: float | None = 25.0) -> dict:
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("Install OpenCV first: pip install opencv-python") from exc

        video = Path(video)
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video}")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_step = max(source_fps / fps_target, 1.0) if fps_target else 1.0
        next_sample = 0.0
        frame_index = 0
        timestamp_offset_ms = self.last_timestamp_ms + 1
        samples: list[np.ndarray] = []
        confidences: list[np.ndarray] = []
        masks: list[np.ndarray] = []

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index + 1e-6 >= next_sample:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timestamp_ms = timestamp_offset_ms + int(round(frame_index * 1000.0 / max(source_fps, 1e-6)))
                    result = self._process_frame(rgb, timestamp_ms, mirrored_source)
                    self.last_timestamp_ms = timestamp_ms
                    samples.append(result["keypoints"])
                    confidences.append(result["confidence"])
                    masks.append(result["valid_mask"])
                    next_sample += sample_step
                frame_index += 1
        finally:
            capture.release()

        if not samples:
            raise RuntimeError(f"No frames extracted from {video}")
        effective_fps = min(source_fps, fps_target) if fps_target else source_fps
        return {
            "keypoints": np.stack(samples).astype(np.float32),
            "confidence": np.stack(confidences).astype(np.float32),
            "valid_mask": np.stack(masks).astype(bool),
            "fps": np.float32(effective_fps),
            "source_fps": np.float32(source_fps),
            "source_frames": np.int32(source_frames),
            "width": np.int32(width),
            "height": np.int32(height),
            "topology_name": "mediapipe_holistic_sign_relevant",
            "source_video": str(video),
            "mirrored_source": np.bool_(mirrored_source),
        }

    def close(self) -> None:
        self.pose.close()
        self.hand.close()
        self.face.close()

    def __enter__(self) -> "MediaPipeTasksExtractor":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
