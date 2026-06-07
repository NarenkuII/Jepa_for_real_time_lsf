from src.text.tokenizer import CharacterTokenizer


def test_character_tokenizer_supports_spelling_and_french():
    tokenizer = CharacterTokenizer()
    tokenizer.train(["ABC", "Bonjour à tous !"])
    assert tokenizer.decode(tokenizer.encode("ABC")) == "ABC"
    assert tokenizer.decode(tokenizer.encode("Bonjour à tous !")) == "Bonjour à tous!"
