from src.data.manifest import write_jsonl
from src.data.synthetic import make_synthetic_sequence
from src.data.dataset_skeleton_text import SkeletonTextDataset
from src.text.tokenizer import WhitespaceTokenizer


def test_skeleton_text_dataset(tmp_path):
    kp, conf, mask = make_synthetic_sequence(0)
    npz = tmp_path / "a.npz"
    import numpy as np

    np.savez_compressed(npz, keypoints=kp, confidence=conf, valid_mask=mask, fps=25.0, topology_name="synthetic", source_video="")
    manifest = tmp_path / "m.jsonl"
    write_jsonl(manifest, [{"id": "a", "keypoints": str(npz), "text_fr": "Bonjour"}])
    tok = WhitespaceTokenizer()
    tok.train(["Bonjour"])
    ds = SkeletonTextDataset(str(manifest), tok)
    assert len(ds) == 1
    assert ds[0]["tokens"][0] == tok.bos_id


def test_skeleton_text_dataset_resamples_fps(tmp_path):
    import json
    import numpy as np

    npz = tmp_path / "sample_10fps.npz"
    keypoints = np.zeros((10, 2, 3), dtype=np.float32)
    keypoints[:, :, 0] = np.arange(10)[:, None]
    np.savez_compressed(npz, keypoints=keypoints, fps=np.float32(10.0))
    manifest = tmp_path / "manifest_10fps.jsonl"
    manifest.write_text(
        json.dumps({"id": "sample", "keypoints": str(npz), "text_fr": "bonjour"}) + "\n",
        encoding="utf-8",
    )

    item = SkeletonTextDataset(str(manifest), target_fps=25.0)[0]
    assert item["keypoints"].shape[0] == 25
    assert np.isclose(item["keypoints"][-1, 0, 0], 9.0)

