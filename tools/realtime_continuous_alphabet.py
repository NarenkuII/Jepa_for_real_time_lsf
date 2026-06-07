from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.keypoints.canonical import NUM_FEATURES, NUM_JOINTS, mediapipe_to_canonical
from src.keypoints.mediapipe_tasks import MediaPipeTasksExtractor
from src.models.continuous_ctc import ContinuousAlphabetCTC, ctc_greedy_decode, ctc_ids_to_text
from src.training.pretrain_jepa import build_model_from_config
from tools.realtime_alphabet import draw_raw_overlay, put_text


def load_model(path: Path, device: torch.device) -> ContinuousAlphabetCTC:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    jepa = build_model_from_config(config, NUM_JOINTS, NUM_FEATURES)
    model = ContinuousAlphabetCTC(jepa.context_encoder, int(config["jepa"]["d_model"]))
    model.load_state_dict(checkpoint["model"])
    return model.eval().to(device)


def available_cameras(cv2, maximum: int = 10) -> list[int]:
    found = []
    for index in range(maximum):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if capture.isOpened():
            ok, _ = capture.read()
            if ok:
                found.append(index)
        capture.release()
    return found


def canonical_segment(raw_frames: deque[dict]) -> np.ndarray:
    return mediapipe_to_canonical(
        np.stack([frame["keypoints"] for frame in raw_frames]),
        np.stack([frame["confidence"] for frame in raw_frames]),
        np.stack([frame["valid_mask"] for frame in raw_frames]),
    )


@torch.inference_mode()
def decode(model, raw_frames: deque[dict], device: torch.device) -> tuple[str, float]:
    keypoints = canonical_segment(raw_frames)
    x = torch.from_numpy(keypoints).unsqueeze(0).to(device)
    mask = torch.ones((1, len(keypoints)), dtype=torch.bool, device=device)
    logits = model(x, mask).float().cpu()
    ids = ctc_greedy_decode(logits, torch.tensor([len(keypoints)]))[0]
    probability = logits.softmax(-1)
    non_blank = 1.0 - probability[..., 0]
    return ctc_ids_to_text(ids), float(non_blank.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous realtime A-Z recognition with a CTC checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/alphabet_continuous_ctc/best.pt"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", choices=("dshow", "msmf", "auto"), default="dshow")
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--delegate", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--predict-every", type=int, default=3)
    parser.add_argument("--min-frames", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=360)
    parser.add_argument("--end-silence-frames", type=int, default=35)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install webcam dependencies: pip install opencv-python mediapipe") from exc

    if args.list_cameras:
        print({"camera_indices": available_cameras(cv2)})
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    extractor = MediaPipeTasksExtractor.create_with_fallback("checkpoints/mediapipe", args.delegate)
    backend = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "auto": cv2.CAP_ANY}[args.backend]
    capture = cv2.VideoCapture(args.camera, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        extractor.close()
        raise RuntimeError(f"Cannot open camera {args.camera}; run again with --list-cameras.")

    frames: deque[dict] = deque(maxlen=args.max_frames)
    active = False
    silence = 0
    prediction = ""
    completed = ""
    confidence = 0.0
    frame_index = 0
    fps = 0.0
    last_time = time.perf_counter()
    print(f"Continuous webcam: torch={device}, MediaPipe={extractor.delegate_name}. SPACE reset, q/ESC quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = max(extractor.last_timestamp_ms + 1, int(time.perf_counter() * 1000))
            raw = extractor._process_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp_ms, mirrored_source=False)
            extractor.last_timestamp_ms = timestamp_ms
            frame_index += 1

            hand_indices = (
                *extractor.topology.groups["left_hand"].indices,
                *extractor.topology.groups["right_hand"].indices,
            )
            hands_visible = bool(raw["valid_mask"][list(hand_indices)].any())
            if hands_visible:
                active = True
                silence = 0
            elif active:
                silence += 1
            if active:
                frames.append(raw)

            if active and len(frames) >= args.min_frames and frame_index % args.predict_every == 0:
                prediction, confidence = decode(model, frames, device)
            if active and silence >= args.end_silence_frames:
                completed = prediction
                frames.clear()
                active = False
                silence = 0
                prediction = ""
                confidence = 0.0

            draw_raw_overlay(cv2, frame, raw, extractor)
            now = time.perf_counter()
            current_fps = 1.0 / max(now - last_time, 1e-6)
            fps = current_fps if fps == 0 else 0.9 * fps + 0.1 * current_fps
            last_time = now
            display = cv2.flip(frame, 1)
            cv2.rectangle(display, (14, 14), (760, 185), (18, 18, 18), -1)
            put_text(cv2, display, f"LIVE: {prediction or '...'}", (28, 62), 1.15, (70, 240, 90), 3)
            put_text(cv2, display, f"LAST: {completed or '...'}", (28, 105), 0.85, (80, 220, 255), 2)
            status = "recording" if active else "show the first letter"
            put_text(
                cv2,
                display,
                f"{status} | {len(frames)} frames | activity {confidence:.0%}",
                (28, 145),
                0.62,
                (220, 220, 220),
            )
            put_text(
                cv2,
                display,
                f"{fps:.1f} FPS | torch {device} | MediaPipe {extractor.delegate_name} | SPACE reset",
                (18, display.shape[0] - 18),
                0.55,
                (80, 255, 80),
            )
            cv2.imshow("Continuous alphabet CTC", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                frames.clear()
                active = False
                silence = 0
                prediction = ""
                completed = ""
                confidence = 0.0
    finally:
        capture.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
