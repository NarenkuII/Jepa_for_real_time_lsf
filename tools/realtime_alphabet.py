from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.keypoints.canonical import GROUPS, NUM_FEATURES, NUM_JOINTS, mediapipe_to_canonical
from src.keypoints.mediapipe_tasks import MediaPipeTasksExtractor
from src.models.alphabet_classifier import AlphabetClassifier, LABELS
from src.training.pretrain_jepa import build_model_from_config


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[AlphabetClassifier, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    jepa = build_model_from_config(config, NUM_JOINTS, NUM_FEATURES)
    model = AlphabetClassifier(jepa.context_encoder, int(config["jepa"]["d_model"]))
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    return model, int(config["data"]["window_size"])


def draw_raw_overlay(cv2, frame: np.ndarray, raw: dict, extractor: MediaPipeTasksExtractor) -> None:
    height, width = frame.shape[:2]
    points = raw["keypoints"][:, :2]
    valid = raw["valid_mask"]
    pixels = np.stack(
        (
            np.clip(points[:, 0] * width, 0, width - 1),
            np.clip(points[:, 1] * height, 0, height - 1),
        ),
        axis=-1,
    ).astype(np.int32)
    for a, b in extractor.topology.edges:
        if valid[a] and valid[b]:
            cv2.line(frame, tuple(pixels[a]), tuple(pixels[b]), (60, 220, 80), 2, cv2.LINE_AA)
    colors = {"pose": (255, 150, 50), "face": (40, 200, 255), "left_hand": (80, 255, 80), "right_hand": (80, 255, 80)}
    for name, color in colors.items():
        radius = 2 if name == "face" else 3
        for index in extractor.topology.groups[name].indices:
            if valid[index]:
                cv2.circle(frame, tuple(pixels[index]), radius, color, -1, cv2.LINE_AA)


def prepare_window(raw_frames: deque[dict], window_size: int) -> tuple[np.ndarray, np.ndarray]:
    raw_keypoints = np.stack([frame["keypoints"] for frame in raw_frames])
    confidence = np.stack([frame["confidence"] for frame in raw_frames])
    valid = np.stack([frame["valid_mask"] for frame in raw_frames])
    canonical = mediapipe_to_canonical(raw_keypoints, confidence, valid)
    frames = canonical.shape[0]
    window = np.zeros((window_size, NUM_JOINTS, NUM_FEATURES), dtype=np.float32)
    padding = np.zeros(window_size, dtype=bool)
    used = min(frames, window_size)
    window[-used:] = canonical[-used:]
    padding[-used:] = True
    return window, padding


@torch.inference_mode()
def predict(model, window: np.ndarray, padding: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(window).unsqueeze(0).to(device)
    mask = torch.from_numpy(padding).unsqueeze(0).to(device)
    return torch.softmax(model(x, mask), dim=-1)[0].float().cpu().numpy()


def put_text(cv2, frame, text: str, position: tuple[int, int], scale=0.65, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime A-Z recognition from a webcam.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/alphabet_graph_jepa_context_fix/best.pt"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--delegate", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--min-frames", type=int, default=24)
    parser.add_argument("--predict-every", type=int, default=3)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install webcam dependencies: pip install opencv-python mediapipe") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, window_size = load_model(args.checkpoint, device)
    extractor = MediaPipeTasksExtractor.create_with_fallback("checkpoints/mediapipe", args.delegate)
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        extractor.close()
        raise RuntimeError(f"Cannot open camera {args.camera}")

    frames: deque[dict] = deque(maxlen=window_size)
    probability_history: deque[np.ndarray] = deque(maxlen=7)
    prediction = np.full(len(LABELS), 1.0 / len(LABELS), dtype=np.float32)
    last_time = time.perf_counter()
    fps = 0.0
    frame_index = 0
    no_hand_frames = 0
    print(f"Alphabet webcam: torch={device}, MediaPipe={extractor.delegate_name}. q/ESC quit, SPACE reset.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = max(extractor.last_timestamp_ms + 1, int(time.perf_counter() * 1000))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw = extractor._process_frame(rgb, timestamp_ms, mirrored_source=False)
            extractor.last_timestamp_ms = timestamp_ms
            frames.append(raw)
            frame_index += 1

            left_start = extractor.topology.groups["left_hand"].indices[0]
            right_start = extractor.topology.groups["right_hand"].indices[0]
            hands_visible = bool(raw["valid_mask"][left_start:].any())
            no_hand_frames = 0 if hands_visible else no_hand_frames + 1
            if no_hand_frames > 15:
                frames.clear()
                probability_history.clear()

            if len(frames) >= args.min_frames and frame_index % args.predict_every == 0:
                window, padding = prepare_window(frames, window_size)
                probability_history.append(predict(model, window, padding, device))
                prediction = np.mean(probability_history, axis=0)

            draw_raw_overlay(cv2, frame, raw, extractor)
            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_time, 1e-6)
            fps = instant_fps if fps == 0 else fps * 0.9 + instant_fps * 0.1
            last_time = now

            top = np.argsort(prediction)[::-1][:3]
            ready = len(frames) >= args.min_frames and hands_visible
            label = LABELS[top[0]] if ready else "..."
            confidence = float(prediction[top[0]]) if ready else 0.0
            color = (70, 240, 90) if confidence >= 0.55 else (40, 210, 255)
            # Mirror only the camera pixels and skeleton. Inference keeps
            # anatomical left/right labels, and text remains readable.
            display = cv2.flip(frame, 1)
            cv2.rectangle(display, (14, 14), (430, 170), (18, 18, 18), -1)
            put_text(cv2, display, f"LETTER: {label}", (28, 68), 1.45, color, 3)
            put_text(cv2, display, f"confidence {confidence:.0%} | buffer {len(frames)}/{window_size}", (28, 104), 0.62)
            if ready:
                ranking = "  ".join(f"{LABELS[i]} {prediction[i]:.0%}" for i in top)
                put_text(cv2, display, ranking, (28, 136), 0.62, (210, 210, 210))
            else:
                put_text(cv2, display, "Show one letter for about 2 seconds", (28, 136), 0.58, (210, 210, 210))
            put_text(cv2, display, f"{fps:.1f} FPS | {device} | SPACE reset | Q quit", (18, display.shape[0] - 18), 0.55, (80, 255, 80))
            cv2.imshow("Realtime alphabet recognition", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                frames.clear()
                probability_history.clear()
                prediction.fill(1.0 / len(LABELS))
    finally:
        capture.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
