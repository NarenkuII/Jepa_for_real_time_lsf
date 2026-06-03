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

