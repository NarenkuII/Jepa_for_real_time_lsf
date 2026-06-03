from __future__ import annotations

from typing import Any

import numpy as np


class KeypointTopologyAdapter:
    def to_internal_format(self, raw_keypoints: Any) -> dict:
        raise NotImplementedError


class NpzKeypointAdapter(KeypointTopologyAdapter):
    def to_internal_format(self, raw_keypoints: str | dict) -> dict:
        if isinstance(raw_keypoints, dict):
            data = raw_keypoints
        else:
            data = dict(np.load(raw_keypoints, allow_pickle=True))
        keypoints = np.asarray(data["keypoints"], dtype=np.float32)
        confidence = np.asarray(data.get("confidence", keypoints[..., 3] if keypoints.shape[-1] > 3 else np.ones(keypoints.shape[:2])), dtype=np.float32)
        valid_mask = np.asarray(data.get("valid_mask", confidence > 0), dtype=bool)
        return {
            "keypoints": keypoints,
            "confidence": confidence,
            "valid_mask": valid_mask,
            "fps": float(np.asarray(data.get("fps", 25.0))),
            "topology_name": str(np.asarray(data.get("topology_name", "custom"))),
            "source_video": str(np.asarray(data.get("source_video", ""))),
        }


class CustomArrayAdapter(KeypointTopologyAdapter):
    def to_internal_format(self, raw_keypoints: Any) -> dict:
        keypoints = np.asarray(raw_keypoints, dtype=np.float32)
        confidence = keypoints[..., 3] if keypoints.shape[-1] > 3 else np.ones(keypoints.shape[:2], dtype=np.float32)
        valid_mask = np.isfinite(keypoints[..., 0]) & (confidence > 0)
        return {"keypoints": keypoints, "confidence": confidence.astype(np.float32), "valid_mask": valid_mask, "fps": 25.0, "topology_name": "custom", "source_video": ""}

