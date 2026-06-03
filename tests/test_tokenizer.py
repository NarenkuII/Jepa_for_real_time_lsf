from src.text.tokenizer import WhitespaceTokenizer


def test_tokenizer_roundtrip(tmp_path):
    tok = WhitespaceTokenizer()
    tok.train(["Demain je vais", "Merci beaucoup"])
    ids = tok.encode("Demain je vais")
    assert ids[0] == tok.bos_id
    path = tmp_path / "vocab.json"
    tok.save(path)
    loaded = WhitespaceTokenizer.load(path)
    assert loaded.decode(ids) == "Demain je vais"

