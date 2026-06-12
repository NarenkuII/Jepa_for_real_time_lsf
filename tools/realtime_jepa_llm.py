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
from src.models.jepa_llm import JepaLlmPrefix
from src.training.pretrain_jepa import build_model_from_config
from tools.realtime_alphabet import draw_raw_overlay, put_text
from tools.realtime_continuous_alphabet import available_cameras


def load_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the optional dependency: pip install -e .[llm]") from exc
    return AutoModelForCausalLM, AutoTokenizer


def load_model(path: Path, device: torch.device):
    AutoModelForCausalLM, AutoTokenizer = load_transformers()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    llm_name = checkpoint["llm_name"]
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=dtype)
    config = checkpoint["jepa_config"]
    jepa = build_model_from_config(config, NUM_JOINTS, NUM_FEATURES)
    model = JepaLlmPrefix(
        jepa.context_encoder,
        int(config["jepa"]["d_model"]),
        llm,
        prefix_tokens=int(checkpoint["prefix_tokens"]),
        resampler_heads=int(checkpoint["resampler_heads"]),
        freeze_llm=True,
    )
    model.load_adapter_state_dict(checkpoint["adapter"])
    return model.eval().to(device), tokenizer


def canonical_segment(raw_frames: deque[dict]) -> np.ndarray:
    return mediapipe_to_canonical(
        np.stack([frame["keypoints"] for frame in raw_frames]),
        np.stack([frame["confidence"] for frame in raw_frames]),
        np.stack([frame["valid_mask"] for frame in raw_frames]),
    )


@torch.inference_mode()
def translate(model, tokenizer, sequence: np.ndarray, device: torch.device, max_new_tokens: int) -> str:
    keypoints = torch.from_numpy(sequence).unsqueeze(0).to(device)
    mask = torch.ones((1, len(sequence)), dtype=torch.bool, device=device)
    start_id = tokenizer.bos_token_id
    if start_id is None:
        start_id = tokenizer.eos_token_id
    if start_id is None:
        raise ValueError("The tokenizer needs a BOS or EOS token for generation.")
    ids = model.greedy_generate(
        keypoints,
        mask,
        start_token_id=int(start_id),
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )
    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmented realtime LSF-to-French JEPA/LLM experiment.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/jepa_llm/best_adapter.pt"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", choices=("dshow", "msmf", "auto"), default="dshow")
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--delegate", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--end-silence-frames", type=int, default=35)
    parser.add_argument("--live-every", type=int, default=0, help="Generate an interim translation every N frames; 0 disables it.")
    parser.add_argument("--max-new-tokens", type=int, default=48)
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
    model, tokenizer = load_model(args.checkpoint, device)
    extractor = MediaPipeTasksExtractor.create_with_fallback("checkpoints/mediapipe", args.delegate)
    backend = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "auto": cv2.CAP_ANY}[args.backend]
    capture = cv2.VideoCapture(args.camera, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        extractor.close()
        raise RuntimeError(f"Cannot open camera {args.camera}; run with --list-cameras.")

    frames: deque[dict] = deque(maxlen=args.max_frames)
    active = False
    silence = 0
    translation = ""
    status = "waiting"
    frame_index = 0
    print(f"JEPA/LLM webcam: torch={device}, MediaPipe={extractor.delegate_name}. SPACE reset, q/ESC quit.")
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
                status = "recording"
            elif active:
                silence += 1
            if active:
                frames.append(raw)

            interim_due = args.live_every > 0 and frame_index % args.live_every == 0
            final_due = active and silence >= args.end_silence_frames
            if len(frames) >= args.min_frames and (interim_due or final_due):
                status = "generating"
                translation = translate(model, tokenizer, canonical_segment(frames), device, args.max_new_tokens)
                status = "done" if final_due else "recording"
            if final_due:
                frames.clear()
                active = False
                silence = 0

            draw_raw_overlay(cv2, frame, raw, extractor)
            display = cv2.flip(frame, 1)
            cv2.rectangle(display, (14, 14), (display.shape[1] - 14, 165), (18, 18, 18), -1)
            put_text(cv2, display, f"FR: {translation or '...'}", (28, 65), 0.9, (70, 240, 90), 2)
            put_text(
                cv2,
                display,
                f"{status} | {len(frames)} frames | torch {device} | MediaPipe {extractor.delegate_name}",
                (28, 112),
                0.62,
                (220, 220, 220),
            )
            put_text(cv2, display, "SPACE reset | Q quit", (28, 147), 0.55, (80, 220, 255))
            cv2.imshow("JEPA to LLM realtime experiment", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                frames.clear()
                active = False
                silence = 0
                translation = ""
                status = "waiting"
    finally:
        capture.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
