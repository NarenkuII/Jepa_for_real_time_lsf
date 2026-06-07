from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.direct_text import load_direct_text_model, predict_direct_text
from src.keypoints.canonical import mediapipe_to_canonical
from src.keypoints.mediapipe_tasks import MediaPipeTasksExtractor
from tools.realtime_alphabet import draw_raw_overlay, put_text


def canonical_segment(frames: deque[dict], max_frames: int) -> np.ndarray:
    sequence = mediapipe_to_canonical(
        np.stack([frame["keypoints"] for frame in frames]),
        np.stack([frame["confidence"] for frame in frames]),
        np.stack([frame["valid_mask"] for frame in frames]),
    )
    if len(sequence) <= max_frames:
        return sequence
    indices = np.linspace(0, len(sequence) - 1, max_frames).round().astype(int)
    return sequence[indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime direct keypoints-to-text without CTC or glosses.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mixed_direct_text/best.pt"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", choices=("dshow", "msmf", "auto"), default="dshow")
    parser.add_argument("--delegate", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--end-silence-frames", type=int, default=35)
    parser.add_argument("--max-text-length", type=int, default=384)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install webcam dependencies: pip install -e .[vision]") from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_direct_text_model(args.checkpoint, device)
    extractor = MediaPipeTasksExtractor.create_with_fallback("checkpoints/mediapipe", args.delegate)
    backend = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "auto": cv2.CAP_ANY}[args.backend]
    capture = cv2.VideoCapture(args.camera, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        extractor.close()
        raise RuntimeError(f"Cannot open camera {args.camera}")

    frames: deque[dict] = deque(maxlen=args.max_frames * 2)
    active = False
    silence = 0
    prediction = ""
    status = "waiting"
    print(f"Direct text webcam: torch={device}, MediaPipe={extractor.delegate_name}. SPACE reset, q/ESC quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = max(extractor.last_timestamp_ms + 1, int(time.perf_counter() * 1000))
            raw = extractor._process_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp, mirrored_source=False)
            extractor.last_timestamp_ms = timestamp
            hand_indices = (
                *extractor.topology.groups["left_hand"].indices,
                *extractor.topology.groups["right_hand"].indices,
            )
            hands_visible = bool(raw["valid_mask"][list(hand_indices)].any())
            if hands_visible:
                active = True
                silence = 0
                status = "recording"
            elif active:
                silence += 1
            if active:
                frames.append(raw)
            if active and silence >= args.end_silence_frames:
                if len(frames) >= args.min_frames:
                    status = "decoding"
                    prediction = predict_direct_text(
                        model,
                        tokenizer,
                        canonical_segment(frames, args.max_frames),
                        device,
                        args.max_text_length,
                    )
                    status = "done"
                frames.clear()
                active = False
                silence = 0

            draw_raw_overlay(cv2, frame, raw, extractor)
            display = cv2.flip(frame, 1)
            cv2.rectangle(display, (14, 14), (display.shape[1] - 14, 160), (18, 18, 18), -1)
            put_text(cv2, display, f"TEXT: {prediction or '...'}", (28, 65), 0.9, (70, 240, 90), 2)
            put_text(
                cv2,
                display,
                f"{status} | {len(frames)} frames | torch {device} | MediaPipe {extractor.delegate_name}",
                (28, 112),
                0.62,
                (220, 220, 220),
            )
            put_text(cv2, display, "SPACE reset | Q quit", (28, 145), 0.55, (80, 220, 255))
            cv2.imshow("Direct JEPA to text", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                frames.clear()
                active = False
                silence = 0
                prediction = ""
                status = "waiting"
    finally:
        capture.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
