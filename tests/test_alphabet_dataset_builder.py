from pathlib import Path

from tools.build_alphabet_dataset import parse_filename, split_for_signer


def test_parse_alphabet_filename_repetitions_and_flags():
    parsed = parse_filename(Path("Dalyan_prise1Q2_statique.mp4"))
    assert parsed["letter"] == "Q"
    assert parsed["repeat"] == 2
    assert parsed["signer_id"] == "dalyan"
    assert parsed["quality_flags"] == ["statique"]


def test_parse_corrected_accented_filename():
    parsed = parse_filename(Path("François_prise1_corrigéD2_petit_doigt_bof.mp4"))
    assert parsed["letter"] == "D"
    assert parsed["repeat"] == 2
    assert parsed["signer_id"] == "francois"
    assert parsed["needs_review_from_name"]


def test_split_is_signer_disjoint():
    assert split_for_signer("marius") == "train"
    assert split_for_signer("thibault") == "val"
    assert split_for_signer("dalyan") == "test"

