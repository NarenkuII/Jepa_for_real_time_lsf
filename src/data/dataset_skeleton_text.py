from __future__ import annotations

import numpy as np

from src.data.manifest import read_jsonl
from src.keypoints.canonical import GROUPS

class SkeletonTextDataset:
    def __init__(self, manifest: str, tokenizer=None, max_text_length: int = 128, target_fps: float | None = None, augment: bool = False):
        self.rows = read_jsonl(manifest)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.target_fps = target_fps
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def _rotate_points_2d(self, pts: np.ndarray, angle_rad: float) -> np.ndarray:
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)
        rot = np.array([[cos_val, -sin_val], [sin_val, cos_val]], dtype=np.float32)
        shape = pts.shape
        return (pts.reshape(-1, 2) @ rot.T).reshape(shape)

    def _rotate_joint_group(self, augmented: np.ndarray, indices: list[int], center_xy: np.ndarray | None, angle_rad: float) -> None:
        for feat_idx in [0, 2, 4, 6]:
            if feat_idx in [0, 2]:
                if center_xy is not None:
                    pts = augmented[:, indices, feat_idx:feat_idx+2] - center_xy[:, None, :]
                    augmented[:, indices, feat_idx:feat_idx+2] = self._rotate_points_2d(pts, angle_rad) + center_xy[:, None, :]
                else:
                    pts = augmented[:, indices, feat_idx:feat_idx+2]
                    augmented[:, indices, feat_idx:feat_idx+2] = self._rotate_points_2d(pts, angle_rad)
            else:
                pts = augmented[:, indices, feat_idx:feat_idx+2]
                augmented[:, indices, feat_idx:feat_idx+2] = self._rotate_points_2d(pts, angle_rad)

    def _augment_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        augmented = keypoints.copy()
        valid_mask = augmented[..., 9:10] > 0.5
        
        # 1. Selective Gaussian Noise
        # Body/Arms: std = 0.02
        body_slice = slice(GROUPS.body.start, GROUPS.body.stop)
        noise_body = np.random.normal(0, 0.02, size=(augmented.shape[0], GROUPS.body.stop - GROUPS.body.start, 4))
        augmented[:, body_slice, :4] += noise_body * valid_mask[:, body_slice]
        
        # Hands: std = 0.006
        hands_slice = slice(GROUPS.left_hand.start, GROUPS.right_hand.stop)
        noise_hands = np.random.normal(0, 0.006, size=(augmented.shape[0], GROUPS.right_hand.stop - GROUPS.left_hand.start, 4))
        augmented[:, hands_slice, :4] += noise_hands * valid_mask[:, hands_slice]
        
        # Face: std = 0.0015
        face_slice = slice(GROUPS.face.start, GROUPS.face.stop)
        noise_face = np.random.normal(0, 0.0015, size=(augmented.shape[0], GROUPS.face.stop - GROUPS.face.start, 4))
        augmented[:, face_slice, :4] += noise_face * valid_mask[:, face_slice]
        
        # 2. Neck Sway (head translation relative to body)
        shift_x = np.random.uniform(-0.03, 0.03)
        shift_y = np.random.uniform(-0.02, 0.02)
        augmented[:, GROUPS.face, :2] += np.array([shift_x, shift_y], dtype=np.float32)
        
        # 3. Hierarchical 2D Rotations
        # Body (shoulders/hips) max 2 degrees
        angle_b = np.radians(np.random.uniform(-2.0, 2.0))
        self._rotate_joint_group(augmented, list(range(GROUPS.body.start, GROUPS.body.stop)), None, angle_b)
        
        # Left arm (elbow=2, wrist=4) around shoulder (0) max 8 degrees
        angle_la = np.radians(np.random.uniform(-8.0, 8.0))
        l_shoulder = augmented[:, 0, :2].copy()
        self._rotate_joint_group(augmented, [2, 4], l_shoulder, angle_la)
        
        # Left wrist (4) around elbow (2) max 10 degrees
        angle_lw = np.radians(np.random.uniform(-10.0, 10.0))
        l_elbow = augmented[:, 2, :2].copy()
        self._rotate_joint_group(augmented, [4], l_elbow, angle_lw)
        
        # Right arm (elbow=3, wrist=5) around shoulder (1) max 8 degrees
        angle_ra = np.radians(np.random.uniform(-8.0, 8.0))
        r_shoulder = augmented[:, 1, :2].copy()
        self._rotate_joint_group(augmented, [3, 5], r_shoulder, angle_ra)
        
        # Right wrist (5) around elbow (3) max 10 degrees
        angle_rw = np.radians(np.random.uniform(-10.0, 10.0))
        r_elbow = augmented[:, 3, :2].copy()
        self._rotate_joint_group(augmented, [5], r_elbow, angle_rw)
        
        # Left hand max 12 degrees around left wrist (joint 4)
        angle_lh = np.radians(np.random.uniform(-12.0, 12.0))
        l_wrist = augmented[:, 4, :2].copy()
        self._rotate_joint_group(augmented, list(range(GROUPS.left_hand.start, GROUPS.left_hand.stop)), l_wrist, angle_lh)
        
        # Right hand max 12 degrees around right wrist (joint 5)
        angle_rh = np.radians(np.random.uniform(-12.0, 12.0))
        r_wrist = augmented[:, 5, :2].copy()
        self._rotate_joint_group(augmented, list(range(GROUPS.right_hand.start, GROUPS.right_hand.stop)), r_wrist, angle_rh)
        
        # Face max 3 degrees around nose (GROUPS.face.start + 23)
        angle_f = np.radians(np.random.uniform(-3.0, 3.0))
        nose = augmented[:, GROUPS.face.start + 23, :2].copy()
        self._rotate_joint_group(augmented, list(range(GROUPS.face.start, GROUPS.face.stop)), nose, angle_f)
        
        # 4. Local Scaling (Zoom)
        scale_l = np.random.uniform(0.98, 1.02)
        augmented[:, GROUPS.left_hand, 2:4] *= scale_l
        augmented[:, GROUPS.left_hand, 4:8] *= scale_l
        
        scale_r = np.random.uniform(0.98, 1.02)
        augmented[:, GROUPS.right_hand, 2:4] *= scale_r
        augmented[:, GROUPS.right_hand, 4:8] *= scale_r
        
        scale_f = np.random.uniform(0.98, 1.02)
        augmented[:, GROUPS.face, 2:4] *= scale_f
        augmented[:, GROUPS.face, 4:8] *= scale_f
        
        # 5. Temporal Masking (5% probability to zero out frames)
        num_frames = augmented.shape[0]
        mask = np.random.uniform(0, 1, size=(num_frames,)) < 0.05
        augmented[mask, :, :8] = 0.0
        augmented[mask, :, 9] = 0.0
        
        return augmented

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        arr = np.load(row["keypoints"], allow_pickle=True)
        keypoints = arr["keypoints"].astype("float32")
        source_fps = float(arr["fps"]) if "fps" in arr else float(row.get("fps", 25.0))
        if self.target_fps and source_fps > 0 and abs(source_fps - self.target_fps) > 1e-3:
            target_frames = max(1, int(round(len(keypoints) * self.target_fps / source_fps)))
            positions = np.linspace(0, len(keypoints) - 1, target_frames)
            left = np.floor(positions).astype(int)
            right = np.minimum(left + 1, len(keypoints) - 1)
            weight = (positions - left).astype(np.float32)[:, None, None]
            keypoints = ((1.0 - weight) * keypoints[left] + weight * keypoints[right]).astype(np.float32)
            
        if self.augment:
            keypoints = self._augment_keypoints(keypoints)
            
        text = row.get("text_fr", "")
        item = {"id": row["id"], "keypoints": keypoints, "text": text}
        if self.tokenizer is not None:
            item["tokens"] = self.tokenizer.encode(text, add_special=True, max_length=self.max_text_length)
        return item
