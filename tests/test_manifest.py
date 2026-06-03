from src.data.manifest import read_jsonl, validate_manifest_rows, write_jsonl


def test_manifest_roundtrip(tmp_path):
    path = tmp_path / "m.jsonl"
    rows = [{"id": "a", "video": "v.mp4", "split": "train"}]
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows
    assert validate_manifest_rows(rows) == []

