import numpy as np

from src.data.direct_text import MixedDirectTextDataset, collate_direct_text
from src.data.manifest import write_jsonl
from src.keypoints.canonical import NUM_FEATURES, NUM_JOINTS
from src.text.tokenizer import CharacterTokenizer


def test_direct_dataset_mixes_label_and_phrase_rows(tmp_path):
    keypoints = np.zeros((8, NUM_JOINTS, NUM_FEATURES), dtype=np.float32)
    keypoints[..., -1] = 1.0
    npz = tmp_path / "sample.npz"
    np.savez_compressed(npz, keypoints=keypoints)
    isolated = tmp_path / "isolated.jsonl"
    phrases = tmp_path / "phrases.jsonl"
    write_jsonl(isolated, [{"id": "a", "keypoints": str(npz), "label": "A"}])
    write_jsonl(phrases, [{"id": "p", "keypoints": str(npz), "text_fr": "Bonjour", "source_type": "matignon"}])
    tokenizer = CharacterTokenizer()
    tokenizer.train(["A", "Bonjour"])
    dataset = MixedDirectTextDataset([isolated, phrases], tokenizer)
    batch = collate_direct_text([dataset[0], dataset[1]])
    assert batch["texts"] == ["A", "Bonjour"]
    assert batch["keypoints"].shape[:2] == (2, 8)
    assert dataset.source_counts == {"alphabet_isolated": 1, "matignon": 1}
