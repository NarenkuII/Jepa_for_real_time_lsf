from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from collections import deque
from types import SimpleNamespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.keypoints.topology import KeypointTopology, mediapipe_holistic_topology
from src.preprocessing.normalization import compute_body_frame, normalize_keypoints
from src.utils.config import load_config


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

POSE_HAND_DUPLICATES = {17, 18, 19, 20, 21, 22}
POSE_FACE_DUPLICATES = set(range(0, 11))
BODY_DEBUG_POSE_INDICES = (11, 12, 13, 14, 15, 16, 23, 24)


def require_cv_mediapipe():
    try:
        import cv2
        import mediapipe as mp
    except Exception as exc:
        raise RuntimeError("Install webcam deps first: pip install opencv-python mediapipe") from exc
    return cv2, mp


def safe_float(value, default: float = 1.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def landmarks_to_array(landmarks, count: int, indices: tuple[int, ...] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if indices is None:
        indices = tuple(range(count))
    values = np.zeros((len(indices), 6), dtype=np.float32)
    confidence = np.zeros((len(indices),), dtype=np.float32)
    valid = np.zeros((len(indices),), dtype=bool)
    if landmarks is None:
        return values, confidence, valid
    for out_i, lm_i in enumerate(indices):
        if lm_i >= len(landmarks.landmark):
            continue
        lm = landmarks.landmark[lm_i]
        visibility = safe_float(getattr(lm, "visibility", 1.0), 1.0)
        presence = safe_float(getattr(lm, "presence", visibility), visibility)
        conf = min(visibility, presence)
        values[out_i] = [lm.x, lm.y, lm.z, conf, visibility, 1.0]
        confidence[out_i] = conf
        valid[out_i] = conf > 0
    return values, confidence, valid


def task_landmarks_to_proto(landmarks):
    if not landmarks:
        return None
    return SimpleNamespace(landmark=landmarks)


def download_if_missing(path: Path, url: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name} ...")
    urllib.request.urlretrieve(url, path)


class TasksHolistic:
    MODEL_URLS = {
        "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    }

    def __init__(self, mp, model_dir: str | Path = "checkpoints/mediapipe", delegate: str = "cpu"):
        self.mp = mp
        model_dir = Path(model_dir)
        pose_model = model_dir / "pose_landmarker_full.task"
        hand_model = model_dir / "hand_landmarker.task"
        face_model = model_dir / "face_landmarker.task"
        download_if_missing(pose_model, self.MODEL_URLS["pose"])
        download_if_missing(hand_model, self.MODEL_URLS["hand"])
        download_if_missing(face_model, self.MODEL_URLS["face"])

        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions
        mode = vision.RunningMode.VIDEO
        delegate_enum = base.Delegate.GPU if delegate == "gpu" else base.Delegate.CPU
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=base(model_asset_path=str(pose_model), delegate=delegate_enum), running_mode=mode, num_poses=1)
        )
        self.hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=base(model_asset_path=str(hand_model), delegate=delegate_enum), running_mode=mode, num_hands=2)
        )
        self.face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=base(model_asset_path=str(face_model), delegate=delegate_enum), running_mode=mode, num_faces=1)
        )

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> SimpleNamespace:
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        pose_res = self.pose.detect_for_video(image, timestamp_ms)
        hand_res = self.hand.detect_for_video(image, timestamp_ms)
        face_res = self.face.detect_for_video(image, timestamp_ms)

        left_hand = None
        right_hand = None
        for landmarks, handedness in zip(hand_res.hand_landmarks, hand_res.handedness):
            label = handedness[0].category_name.lower() if handedness else ""
            if label == "left":
                left_hand = task_landmarks_to_proto(landmarks)
            elif label == "right":
                right_hand = task_landmarks_to_proto(landmarks)
            elif left_hand is None:
                left_hand = task_landmarks_to_proto(landmarks)
            else:
                right_hand = task_landmarks_to_proto(landmarks)

        return SimpleNamespace(
            pose_landmarks=task_landmarks_to_proto(pose_res.pose_landmarks[0]) if pose_res.pose_landmarks else None,
            face_landmarks=task_landmarks_to_proto(face_res.face_landmarks[0]) if face_res.face_landmarks else None,
            left_hand_landmarks=left_hand,
            right_hand_landmarks=right_hand,
        )

    def close(self) -> None:
        self.pose.close()
        self.hand.close()
        self.face.close()


def holistic_to_internal(result, full_face: bool) -> tuple[dict, KeypointTopology]:
    face_indices = tuple(range(468)) if full_face else FACE_SIGN_RELEVANT
    topology = mediapipe_holistic_topology("full" if full_face else "sign_relevant")
    parts = []
    confidences = []
    masks = []
    for landmarks, count, indices in (
        (result.pose_landmarks, 33, None),
        (result.face_landmarks, len(face_indices), face_indices),
        (result.left_hand_landmarks, 21, None),
        (result.right_hand_landmarks, 21, None),
    ):
        arr, conf, valid = landmarks_to_array(landmarks, count, indices)
        parts.append(arr)
        confidences.append(conf)
        masks.append(valid)
    keypoints = np.concatenate(parts, axis=0)[None].astype(np.float32)
    confidence = np.concatenate(confidences, axis=0)[None].astype(np.float32)
    valid_mask = np.concatenate(masks, axis=0)[None].astype(bool)
    return {"keypoints": keypoints, "confidence": confidence, "valid_mask": valid_mask, "fps": 0.0, "topology_name": topology.name}, topology


def draw_raw_overlay(cv2, mp, frame, result, draw_face_mesh: bool) -> None:
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    holistic = mp.solutions.holistic
    drawing.draw_landmarks(frame, result.pose_landmarks, holistic.POSE_CONNECTIONS, landmark_drawing_spec=styles.get_default_pose_landmarks_style())
    drawing.draw_landmarks(frame, result.left_hand_landmarks, holistic.HAND_CONNECTIONS, landmark_drawing_spec=styles.get_default_hand_landmarks_style(), connection_drawing_spec=styles.get_default_hand_connections_style())
    drawing.draw_landmarks(frame, result.right_hand_landmarks, holistic.HAND_CONNECTIONS, landmark_drawing_spec=styles.get_default_hand_landmarks_style(), connection_drawing_spec=styles.get_default_hand_connections_style())
    if result.face_landmarks is not None:
        connections = holistic.FACEMESH_TESSELATION if draw_face_mesh else holistic.FACEMESH_CONTOURS
        drawing.draw_landmarks(
            frame,
            result.face_landmarks,
            connections,
            landmark_drawing_spec=None,
            connection_drawing_spec=styles.get_default_face_mesh_tesselation_style() if draw_face_mesh else styles.get_default_face_mesh_contours_style(),
        )


def draw_raw_internal_overlay(cv2, frame, raw: dict, topology: KeypointTopology, full_face: bool) -> None:
    h, w = frame.shape[:2]
    pts = raw["keypoints"][0, :, :2].copy()
    valid = raw["valid_mask"][0]
    pix = np.zeros((len(pts), 2), dtype=np.int32)
    pix[:, 0] = np.clip((pts[:, 0] * w).astype(np.int32), 0, w - 1)
    pix[:, 1] = np.clip((pts[:, 1] * h).astype(np.int32), 0, h - 1)
    for a, b in topology.edges:
        if a < len(pix) and b < len(pix) and valid[a] and valid[b]:
            cv2.line(frame, tuple(pix[a]), tuple(pix[b]), (70, 230, 100), 2, cv2.LINE_AA)
    colors = {
        "pose": (255, 130, 40),
        "face": (40, 210, 255),
        "left_hand": (80, 255, 80),
        "right_hand": (80, 255, 80),
    }
    for name, color in colors.items():
        radius = 1 if name == "face" and full_face else 3
        for idx in topology.groups[name].indices:
            if name == "pose" and idx in POSE_HAND_DUPLICATES | POSE_FACE_DUPLICATES:
                continue
            if idx < len(pix) and valid[idx]:
                cv2.circle(frame, tuple(pix[idx]), radius, color, -1, cv2.LINE_AA)


def draw_text(cv2, image, text: str, x: int, y: int, color=(230, 230, 230), scale: float = 0.48) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_panel_title(cv2, image, title: str, x: int, y: int, w: int, h: int) -> None:
    cv2.rectangle(image, (x, y), (x + w, y + h), (36, 36, 36), 1)
    draw_text(cv2, image, title, x + 10, y + 22, color=(180, 230, 255), scale=0.55)


def draw_points(cv2, canvas, points: np.ndarray, valid: np.ndarray, rect: tuple[int, int, int, int], color, edges=(), scale: float = 1.0) -> None:
    x, y, w, h = rect
    cx = x + w // 2
    cy = y + h // 2
    xy = points[:, :2].copy()
    pix = np.zeros((len(xy), 2), dtype=np.int32)
    pix[:, 0] = (cx + xy[:, 0] * scale).astype(np.int32)
    pix[:, 1] = (cy + xy[:, 1] * scale).astype(np.int32)
    for a, b in edges:
        if a < len(pix) and b < len(pix) and valid[a] and valid[b]:
            cv2.line(canvas, tuple(pix[a]), tuple(pix[b]), color, 1, cv2.LINE_AA)
    for i, p in enumerate(pix):
        if valid[i]:
            cv2.circle(canvas, tuple(p), 2, color, -1, cv2.LINE_AA)


def pose_edges_for_group(topology: KeypointTopology) -> tuple[tuple[int, int], ...]:
    pose_indices = list(BODY_DEBUG_POSE_INDICES)
    local = {global_idx: local_idx for local_idx, global_idx in enumerate(pose_indices)}
    return tuple((local[a], local[b]) for a, b in topology.edges if a in local and b in local)


def seated_friendly_pose(raw: dict, topology: KeypointTopology) -> tuple[np.ndarray, np.ndarray, float, str]:
    xyz = raw["keypoints"][0, :, :3]
    valid = raw["valid_mask"][0]
    lm = topology.landmarks
    left_shoulder = xyz[lm["left_shoulder"]]
    right_shoulder = xyz[lm["right_shoulder"]]
    shoulder_center = (left_shoulder + right_shoulder) * 0.5
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

    hips_ok = bool(valid[lm["left_hip"]] and valid[lm["right_hip"]])
    center = shoulder_center.copy()
    mode = "shoulder-centered"
    if hips_ok:
        hip_center = (xyz[lm["left_hip"]] + xyz[lm["right_hip"]]) * 0.5
        torso_height = np.linalg.norm(shoulder_center - hip_center)
        # When seated/cropped, hip landmarks often jump below the frame. Do not let that destroy the debug view.
        if shoulder_width > 1e-6 and torso_height < shoulder_width * 1.8:
            center = (shoulder_center + hip_center) * 0.5
            mode = "torso-centered"

    scale = max(float(shoulder_width), 1e-4)
    pose_idx = list(BODY_DEBUG_POSE_INDICES)
    pose = (xyz[pose_idx] - center[None, :]) / scale
    return pose, valid[pose_idx], scale, mode


def draw_hand(cv2, canvas, points: np.ndarray, valid: np.ndarray, rect: tuple[int, int, int, int], title: str) -> None:
    x, y, w, h = rect
    draw_panel_title(cv2, canvas, title, x, y, w, h)
    hand_edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    )
    draw_points(cv2, canvas, points, valid, (x + 8, y + 28, w - 16, h - 36), (90, 220, 120), hand_edges, scale=95.0)


def compute_debug_stats(raw: dict, norm: dict, topology: KeypointTopology, frame_dt: float, buffer_len: int) -> dict[str, float]:
    xyz = raw["keypoints"][0, :, :3]
    valid = raw["valid_mask"][0]
    torso_center, body_scale = compute_body_frame(raw["keypoints"][:, :, :3], topology)
    lm = topology.landmarks
    shoulder_width = np.linalg.norm(xyz[lm["left_shoulder"]] - xyz[lm["right_shoulder"]])
    torso_height = np.linalg.norm((xyz[lm["left_shoulder"]] + xyz[lm["right_shoulder"]]) * 0.5 - (xyz[lm["left_hip"]] + xyz[lm["right_hip"]]) * 0.5)
    left_scale = np.linalg.norm(xyz[lm["left_wrist"]] - xyz[lm["left_index_mcp"]])
    right_scale = np.linalg.norm(xyz[lm["right_wrist"]] - xyz[lm["right_index_mcp"]])
    features = norm["keypoints"][-1]
    velocity = features[:, 12:15]
    acceleration = features[:, 15:18]
    groups = topology.groups
    stats = {
        "buffer_frames": float(buffer_len),
        "frame_dt_ms": frame_dt * 1000.0,
        "jepa_shape_t": float(norm["keypoints"].shape[0]),
        "jepa_shape_j": float(raw["keypoints"].shape[1]),
        "jepa_shape_f": float(norm["keypoints"].shape[-1]),
        "valid_ratio": float(valid.mean()),
        "pose_presence": float(valid[list(groups["pose"].indices)].mean()),
        "face_presence": float(valid[list(groups["face"].indices)].mean()),
        "left_hand_presence": float(valid[list(groups["left_hand"].indices)].mean()),
        "right_hand_presence": float(valid[list(groups["right_hand"].indices)].mean()),
        "body_scale": float(body_scale[0, 0]),
        "shoulder_width": float(shoulder_width),
        "torso_height": float(torso_height),
        "left_hand_scale": float(left_scale),
        "right_hand_scale": float(right_scale),
        "mean_velocity": float(np.linalg.norm(velocity, axis=-1).mean()),
        "max_velocity": float(np.linalg.norm(velocity, axis=-1).max()),
        "mean_acceleration": float(np.linalg.norm(acceleration, axis=-1).mean()),
        "torso_center_x": float(torso_center[0, 0]),
        "torso_center_y": float(torso_center[0, 1]),
        "face_joints": float(len(topology.groups["face"].indices)),
    }
    return stats


def make_debug_canvas(cv2, raw: dict, norm: dict, topology: KeypointTopology, stats: dict[str, float], fps: float) -> np.ndarray:
    canvas = np.zeros((820, 1400, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)
    raw_valid = raw["valid_mask"][0]
    norm_frame = norm["keypoints"][-1]
    face_rel = norm_frame[:, 6:9]
    hand_rel = norm_frame[:, 9:12]

    pose, pose_valid, pose_scale, pose_mode = seated_friendly_pose(raw, topology)
    draw_panel_title(cv2, canvas, f"Body pose normalized ({pose_mode}, scale=shoulders)", 20, 20, 500, 360)
    draw_points(cv2, canvas, pose, pose_valid, (35, 58, 470, 300), (80, 190, 255), pose_edges_for_group(topology), scale=135.0)

    draw_panel_title(cv2, canvas, f"Face normalized zoom ({int(stats['face_joints'])} pts, nose/body scale)", 540, 20, 360, 360)
    face_idx = list(topology.groups["face"].indices)
    draw_points(cv2, canvas, face_rel[face_idx], raw_valid[face_idx], (555, 58, 330, 300), (230, 180, 80), (), scale=520.0)

    left_idx = list(topology.groups["left_hand"].indices)
    right_idx = list(topology.groups["right_hand"].indices)
    draw_hand(cv2, canvas, hand_rel[left_idx], raw_valid[left_idx], (20, 410, 420, 360), "Left hand local zoom (wrist centered)")
    draw_hand(cv2, canvas, hand_rel[right_idx], raw_valid[right_idx], (470, 410, 420, 360), "Right hand local zoom (wrist centered)")

    draw_panel_title(cv2, canvas, "JEPA feature/debug info", 930, 20, 440, 750)
    lines = [
        f"FPS: {fps:5.1f}",
        f"JEPA input now: [T={int(stats['jepa_shape_t'])}, J={int(stats['jepa_shape_j'])}, F={int(stats['jepa_shape_f'])}]",
        "F blocks: global/body/face/hand/vel/acc",
        "Debug view separates pose, face and hands.",
        f"pose debug scale: {pose_scale:.4f}",
        f"buffer frames: {int(stats['buffer_frames'])}",
        f"valid ratio: {stats['valid_ratio']:.2f}",
        f"pose presence: {stats['pose_presence']:.2f}",
        f"face presence: {stats['face_presence']:.2f}",
        f"face joints: {int(stats['face_joints'])}",
        f"left hand presence: {stats['left_hand_presence']:.2f}",
        f"right hand presence: {stats['right_hand_presence']:.2f}",
        f"body scale: {stats['body_scale']:.4f}",
        f"shoulder width: {stats['shoulder_width']:.4f}",
        f"torso height: {stats['torso_height']:.4f}",
        f"left hand scale: {stats['left_hand_scale']:.4f}",
        f"right hand scale: {stats['right_hand_scale']:.4f}",
        f"mean velocity: {stats['mean_velocity']:.4f}",
        f"max velocity: {stats['max_velocity']:.4f}",
        f"mean acceleration: {stats['mean_acceleration']:.4f}",
        f"torso center x/y: {stats['torso_center_x']:.3f}, {stats['torso_center_y']:.3f}",
        "",
        "Keys: q/ESC quit, s save frames",
    ]
    y = 62
    for line in lines:
        draw_text(cv2, canvas, line, 948, y, scale=0.45)
        y += 26
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/realtime.yaml")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--model_complexity", type=int, default=1)
    parser.add_argument("--full_face", action="store_true", default=True, help="Use all 468 face landmarks. This is the default.")
    parser.add_argument("--face_subset", action="store_true", help="Use only the small sign-relevant face subset instead of all 468 face points.")
    parser.add_argument("--draw_face_mesh", action="store_true", help="Draw dense face mesh on raw webcam overlay. Default draws contours only.")
    parser.add_argument("--buffer_size", type=int, default=128)
    parser.add_argument("--save_dir", default="reports/webcam_debug")
    parser.add_argument("--backend", choices=["auto", "solutions", "tasks"], default="auto")
    parser.add_argument("--model_dir", default="checkpoints/mediapipe")
    parser.add_argument("--delegate", choices=["cpu", "gpu"], default="gpu", help="MediaPipe Tasks delegate. GPU is attempted by default in tasks backend.")
    args = parser.parse_args()
    full_face = args.full_face and not args.face_subset

    cv2, mp = require_cv_mediapipe()
    cfg = load_config(args.config)
    camera_id = args.camera if args.camera is not None else int(cfg.get("realtime", {}).get("camera_id", 0))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    frame_times = deque(maxlen=30)
    keypoint_buffer = deque(maxlen=args.buffer_size)
    confidence_buffer = deque(maxlen=args.buffer_size)
    valid_buffer = deque(maxlen=args.buffer_size)
    use_solutions = hasattr(mp, "solutions") and args.backend in ("auto", "solutions")
    if args.backend == "solutions" and not hasattr(mp, "solutions"):
        raise RuntimeError("Requested --backend solutions, but this MediaPipe install has no mp.solutions API.")
    if use_solutions:
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=args.model_complexity,
            smooth_landmarks=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        backend_name = "solutions"
    else:
        try:
            holistic = TasksHolistic(mp, args.model_dir, delegate=args.delegate)
        except Exception:
            if args.delegate != "gpu":
                raise
            print("GPU delegate failed for MediaPipe Tasks; falling back to CPU.")
            holistic = TasksHolistic(mp, args.model_dir, delegate="cpu")
        backend_name = "tasks"
    print("Running webcam skeleton debug. Press q or ESC to quit, s to save current windows.")
    print(f"MediaPipe backend: {backend_name}")
    try:
        frame_index = 0
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(frame_index * 1000 / 30)
            result = holistic.process(rgb) if backend_name == "solutions" else holistic.process(rgb, timestamp_ms)
            frame_index += 1
            raw, topology = holistic_to_internal(result, full_face=full_face)
            keypoint_buffer.append(raw["keypoints"][0])
            confidence_buffer.append(raw["confidence"][0])
            valid_buffer.append(raw["valid_mask"][0])
            sequence_payload = {
                "keypoints": np.asarray(keypoint_buffer, dtype=np.float32),
                "confidence": np.asarray(confidence_buffer, dtype=np.float32),
                "valid_mask": np.asarray(valid_buffer, dtype=bool),
                "fps": raw["fps"],
                "topology_name": raw["topology_name"],
            }
            norm = normalize_keypoints(sequence_payload, topology=topology)

            raw_view = frame.copy()
            if backend_name == "solutions":
                draw_raw_overlay(cv2, mp, raw_view, result, draw_face_mesh=args.draw_face_mesh)
            else:
                draw_raw_internal_overlay(cv2, raw_view, raw, topology, full_face)
            dt = time.perf_counter() - t0
            frame_times.append(dt)
            fps = 1.0 / max(np.mean(frame_times), 1e-6)
            stats = compute_debug_stats(raw, norm, topology, dt, len(keypoint_buffer))
            draw_text(cv2, raw_view, f"FPS {fps:.1f} | valid {stats['valid_ratio']:.2f} | hands L/R {stats['left_hand_presence']:.2f}/{stats['right_hand_presence']:.2f}", 20, 35, color=(30, 255, 80), scale=0.7)
            draw_text(cv2, raw_view, "q/ESC quit | s save | raw MediaPipe overlay", 20, 65, color=(30, 255, 80), scale=0.55)
            debug_canvas = make_debug_canvas(cv2, raw, norm, topology, stats, fps)

            cv2.imshow("raw_overlay", raw_view)
            cv2.imshow("normalized_jepa_debug", debug_canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(save_dir / f"raw_overlay_{stamp}.png"), raw_view)
                cv2.imwrite(str(save_dir / f"normalized_jepa_debug_{stamp}.png"), debug_canvas)
                np.savez_compressed(
                    save_dir / f"keypoints_{stamp}.npz",
                    keypoints=np.asarray(keypoint_buffer, dtype=np.float32),
                    confidence_sequence=np.asarray(confidence_buffer, dtype=np.float32),
                    valid_mask_sequence=np.asarray(valid_buffer, dtype=bool),
                    last_frame_keypoints=raw["keypoints"],
                    normalized_sequence=norm["keypoints"],
                    confidence=raw["confidence"],
                    valid_mask=raw["valid_mask"],
                    topology_name=topology.name,
                )
                print(f"Saved debug snapshot to {save_dir}")
    finally:
        holistic.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
