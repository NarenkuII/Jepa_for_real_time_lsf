from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from src.data.manifest import write_jsonl
from src.keypoints.canonical import FEATURE_NAMES, NUM_FEATURES, NUM_JOINTS, mediapipe_to_canonical
from src.keypoints.topology import SIGN_RELEVANT_FACE_LANDMARKS

PATH_COLUMNS = ("keypoints", "keypoint_path", "path", "file", "filename", "mediapipe")
TEXT_COLUMNS = ("text_fr", "text", "subtitle", "sentence", "translation", "caption")
ID_COLUMNS = ("id", "clip_id", "sample_id", "video_id", "name")
SPLIT_COLUMNS = ("split", "partition", "set", "subset")
ARRAY_ALIASES = {
    "pose": ("pose", "pose_landmarks", "pose_keypoints"),
    "face": ("face", "face_landmarks", "face_keypoints"),
    "left_hand": ("left_hand", "left_hand_landmarks", "hand_left"),
    "right_hand": ("right_hand", "right_hand_landmarks", "hand_right"),
}


def _first(row: dict[str, Any], names: tuple[str, ...], default: str = "") -> str:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def infer_split(value: str | Path) -> str:
    parts = [part.lower() for part in Path(value).parts]
    for split in ("train", "val", "validation", "dev", "test"):
        if split in parts:
            return "val" if split in ("validation", "dev") else split
    return "train"


def discover_index(root: str | Path) -> Path | None:
    root = Path(root)
    candidates = []
    for pattern in ("*.jsonl", "*.csv", "*.json"):
        candidates.extend(root.rglob(pattern))
    named = [
        path
        for path in candidates
        if any(token in path.name.lower() for token in ("metadata", "manifest", "index", "subtitle", "caption"))
    ]
    if not named:
        return None
    scored = sorted(
        named,
        key=lambda path: (
            len(path.parts),
            path.name,
        ),
    )
    return scored[0] if scored else None


def read_index(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return payload
    for key in ("items", "samples", "clips", "data"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"Unsupported JSON index structure: {path}")


def _array(payload: Any, aliases: tuple[str, ...]) -> np.ndarray | None:
    keys = set(payload.files) if hasattr(payload, "files") else set(payload)
    for name in aliases:
        if name in keys:
            return np.asarray(payload[name])
    return None


def _landmark_values(array: np.ndarray, expected: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3 or array.shape[1] < expected or array.shape[2] < 2:
        raise ValueError(f"Expected [T, >= {expected}, >= 2], got {array.shape}")
    xy = array[:, :expected, :2]
    confidence = array[:, :expected, 3] if array.shape[2] >= 4 else np.ones(xy.shape[:2], dtype=np.float32)
    valid = np.isfinite(xy).all(axis=-1) & np.isfinite(confidence) & (confidence > 0.0)
    return np.nan_to_num(xy), np.nan_to_num(confidence), valid


def convert_keypoint_file(path: str | Path) -> tuple[np.ndarray, float]:
    path = Path(path)
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray):
        if loaded.dtype == object and loaded.shape == ():
            payload = loaded.item()
        else:
            payload = {"keypoints": loaded}
    else:
        payload = loaded

    try:
        keypoints = _array(payload, ("keypoints", "features"))
        fps_value = _array(payload, ("fps", "frame_rate"))
        fps = float(np.asarray(fps_value).reshape(-1)[0]) if fps_value is not None else 25.0
        if keypoints is not None and keypoints.ndim == 3 and keypoints.shape[1:] == (NUM_JOINTS, NUM_FEATURES):
            return keypoints.astype(np.float32), fps

        raw = _array(payload, ("raw_keypoints",))
        confidence = _array(payload, ("confidence", "scores"))
        valid = _array(payload, ("valid_mask", "valid"))
        if raw is not None and confidence is not None and valid is not None:
            return mediapipe_to_canonical(raw, confidence, valid), fps

        arrays = {name: _array(payload, aliases) for name, aliases in ARRAY_ALIASES.items()}
        if any(value is None for value in arrays.values()):
            missing = [name for name, value in arrays.items() if value is None]
            raise ValueError(f"Missing MediaPipe arrays: {', '.join(missing)}")

        pose_xy, pose_conf, pose_valid = _landmark_values(arrays["pose"], 33)
        face_xy, face_conf, face_valid = _landmark_values(arrays["face"], max(SIGN_RELEVANT_FACE_LANDMARKS) + 1)
        left_xy, left_conf, left_valid = _landmark_values(arrays["left_hand"], 21)
        right_xy, right_conf, right_valid = _landmark_values(arrays["right_hand"], 21)
        face_indices = list(SIGN_RELEVANT_FACE_LANDMARKS)
        raw_xy = np.concatenate((pose_xy, face_xy[:, face_indices], left_xy, right_xy), axis=1)
        raw_conf = np.concatenate((pose_conf, face_conf[:, face_indices], left_conf, right_conf), axis=1)
        raw_valid = np.concatenate((pose_valid, face_valid[:, face_indices], left_valid, right_valid), axis=1)
        raw_xyz = np.zeros((*raw_xy.shape[:-1], 3), dtype=np.float32)
        raw_xyz[..., :2] = raw_xy
        return mediapipe_to_canonical(raw_xyz, raw_conf, raw_valid), fps
    finally:
        if hasattr(loaded, "close"):
            loaded.close()


def _keypoint_files(root: Path) -> list[Path]:
    return sorted(path for pattern in ("*.npz", "*.npy") for path in root.rglob(pattern))


def _official_archive_members(root: Path) -> dict[str, tuple[Path, str]]:
    members: dict[str, tuple[Path, str]] = {}
    for archive in sorted(root.rglob("mediapipe[123].zip")):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".zip"):
                    continue
                members.setdefault(Path(member.filename).stem.lower(), (archive, member.filename))
    return members


def _official_metadata(root: Path) -> list[dict[str, Any]]:
    subtitle_paths = list(root.rglob("subtitles.csv"))
    info_paths = list(root.rglob("info_mediapirgb.csv"))
    if not subtitle_paths or not info_paths:
        return []

    subtitles = read_index(subtitle_paths[0])
    info = read_index(info_paths[0])
    info_by_id = {
        _first(row, ("video_id",) + ID_COLUMNS).lower(): row
        for row in info
        if _first(row, ("video_id",) + ID_COLUMNS)
    }
    merged = []
    for subtitle in subtitles:
        sample_id = _first(subtitle, ID_COLUMNS)
        if not sample_id:
            continue
        info_row = info_by_id.get(sample_id.lower(), {})
        merged.append(
            {
                **subtitle,
                "clip_id": sample_id,
                "text_fr": _first(subtitle, TEXT_COLUMNS),
                "split": _first(info_row, SPLIT_COLUMNS, "train"),
                "fps": _first(info_row, ("fps",), "25"),
            }
        )
    return merged


def _read_landmark_csv(
    archive: zipfile.ZipFile,
    sample_id: str,
    suffix: str,
    landmarks: int,
) -> np.ndarray:
    matching = [
        name
        for name in archive.namelist()
        if name.lower().endswith(f"{sample_id.lower()}_{suffix}.csv")
    ]
    if len(matching) != 1:
        raise ValueError(f"Expected one {suffix} CSV for {sample_id}, found {len(matching)}")
    text = archive.read(matching[0]).decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    output = np.zeros((len(lines), landmarks, 4), dtype=np.float32)
    expected = landmarks * 4
    for row_index, line in enumerate(lines):
        fields = line.rstrip("\t").split("\t")
        if len(fields) <= 1 or not any(fields[1:]):
            continue
        values = np.fromstring("\t".join(fields[1:]), dtype=np.float32, sep="\t")
        if values.size != expected:
            raise ValueError(
                f"Malformed {suffix} row {row_index} for {sample_id}: "
                f"{values.size} values, expected {expected}"
            )
        output[row_index] = values.reshape(landmarks, 4)
    return output


def convert_official_clip(
    outer_archive: Path,
    member_name: str,
    sample_id: str,
) -> np.ndarray:
    with zipfile.ZipFile(outer_archive) as outer:
        payload = io.BytesIO(outer.read(member_name))
    with zipfile.ZipFile(payload) as inner:
        pose = _read_landmark_csv(inner, sample_id, "pose", 33)
        face = _read_landmark_csv(inner, sample_id, "face", 468)
        left = _read_landmark_csv(inner, sample_id, "left_hand", 21)
        right = _read_landmark_csv(inner, sample_id, "right_hand", 21)

    frame_count = min(len(pose), len(face), len(left), len(right))
    if frame_count < 1:
        raise ValueError(f"No frames found for {sample_id}")
    pose = pose[:frame_count]
    face = face[:frame_count, list(SIGN_RELEVANT_FACE_LANDMARKS)]
    left = left[:frame_count]
    right = right[:frame_count]
    arrays = (pose, face, left, right)
    raw_xyz = np.concatenate(tuple(array[..., :3] for array in arrays), axis=1)
    valid = np.concatenate(
        tuple(np.isfinite(array[..., :2]).all(axis=-1) & (np.abs(array[..., :2]).sum(axis=-1) > 0) for array in arrays),
        axis=1,
    )
    confidence = valid.astype(np.float32)
    return mediapipe_to_canonical(raw_xyz, confidence, valid)


def _resolve_source(root: Path, row: dict[str, Any], by_stem: dict[str, list[Path]]) -> Path | None:
    path_value = _first(row, PATH_COLUMNS)
    if path_value:
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.exists():
            return candidate
    identifier = _first(row, ID_COLUMNS)
    matches = by_stem.get(Path(identifier).stem.lower(), [])
    return matches[0] if len(matches) == 1 else None


def prepare_mediapi_rgb(
    source_root: str | Path,
    output_root: str | Path,
    index_path: str | Path | None = None,
    overwrite: bool = False,
    max_samples: int = 0,
) -> dict[str, Any]:
    source_root = Path(source_root)
    output_root = Path(output_root)
    files = _keypoint_files(source_root)
    official_members = _official_archive_members(source_root)
    by_stem: dict[str, list[Path]] = {}
    for path in files:
        by_stem.setdefault(path.stem.lower(), []).append(path)

    discovered = Path(index_path) if index_path else discover_index(source_root)
    metadata = _official_metadata(source_root)
    if not metadata:
        metadata = read_index(discovered) if discovered else []
    entries: list[tuple[Path | tuple[Path, str], dict[str, Any]]] = []
    if metadata and official_members:
        for row in metadata:
            identifier = _first(row, ID_COLUMNS).lower()
            source = official_members.get(identifier)
            if source is not None:
                entries.append((source, row))
    elif metadata:
        for row in metadata:
            source = _resolve_source(source_root, row, by_stem)
            if source is not None:
                entries.append((source, row))
    else:
        entries = [(path, {}) for path in files]
    if max_samples:
        entries = entries[:max_samples]

    rows = []
    errors = []
    for position, (source, metadata_row) in enumerate(entries, start=1):
        source_stem = Path(source[1]).stem if isinstance(source, tuple) else source.stem
        source_description = (
            f"{source[0].resolve()}::{source[1]}" if isinstance(source, tuple) else str(source.resolve())
        )
        sample_id = _safe_id(_first(metadata_row, ID_COLUMNS, source_stem))
        split_default = source_description if isinstance(source, tuple) else str(source.relative_to(source_root))
        split = infer_split(_first(metadata_row, SPLIT_COLUMNS, split_default))
        target = output_root / "keypoints" / split / f"{sample_id}.npz"
        try:
            if overwrite or not target.exists():
                if isinstance(source, tuple):
                    features = convert_official_clip(source[0], source[1], sample_id)
                    fps = float(_first(metadata_row, ("fps",), "25"))
                else:
                    features, fps = convert_keypoint_file(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    target,
                    keypoints=features,
                    valid_mask=features[..., -1].astype(bool),
                    fps=np.float32(fps),
                    topology_name="canonical_sign89",
                    feature_names=np.asarray(FEATURE_NAMES),
                    source_keypoints=source_description,
                )
            else:
                with np.load(target) as payload:
                    features = payload["keypoints"]
                    fps = float(payload.get("fps", 25.0))
            text = _first(metadata_row, TEXT_COLUMNS)
            rows.append(
                {
                    "id": f"mediapi_rgb_{sample_id}",
                    "split": split,
                    "source_type": "mediapi_rgb",
                    "keypoints": str(target.resolve()),
                    "text_fr": text,
                    "frames": int(features.shape[0]),
                    "fps": fps,
                    "num_joints": NUM_JOINTS,
                    "num_features": NUM_FEATURES,
                    "source_keypoints": source_description,
                }
            )
        except Exception as exc:
            errors.append({"source": source_description, "error": str(exc)})
        if position % 250 == 0:
            print(f"[mediapi-rgb] {position}/{len(entries)}", flush=True)

    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        write_jsonl(output_root / "manifests" / f"mediapi_rgb_{split}.jsonl", split_rows)
        write_jsonl(
            output_root / "manifests" / f"mediapi_rgb_text_{split}.jsonl",
            (row for row in split_rows if row["text_fr"]),
        )
    report = {
        "source_root": str(source_root.resolve()),
        "index": str(discovered.resolve()) if discovered else None,
        "keypoint_files_found": len(files) + len(official_members),
        "metadata_rows": len(metadata),
        "converted": len(rows),
        "with_text": sum(bool(row["text_fr"]) for row in rows),
        "splits": {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")},
        "errors": errors,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
