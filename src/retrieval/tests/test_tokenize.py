import pytest

from src.retrieval.tokenize import keyword_overlap_score, tokenize


def test_tokenize_is_deterministic():
    text = "RL\tand\nIL!!!"
    first = tokenize(text)
    second = tokenize(text)

    assert first == second
    assert first == ("rl", "and", "il")


def test_keyword_overlap_f1_scoring():
    assert (
        keyword_overlap_score("reinforcement learning", "reinforcement learning") == 1.0
    )

    partial = keyword_overlap_score("reinforcement learning", "reinforcement")
    assert partial == pytest.approx(2.0 / 3.0)


def test_keyword_overlap_handles_empty_inputs():
    assert keyword_overlap_score("", "reinforcement") == 0.0
    assert keyword_overlap_score("reinforcement", "") == 0.0
