import numpy as np
from src.machine_learning.tfidf import (
    tokenize,
    compute_tf,
    compute_idf,
    compute_tfidf,
)

def test_tokenize():
    doc = "Hello world hello"
    tokens = tokenize(doc)
    assert tokens == ["hello", "world", "hello"]


def test_compute_tf():
    tokens = ["a", "b", "a", "c"]
    tf = compute_tf(tokens)
    assert tf["a"] == 0.5
    assert tf["b"] == 0.25
    assert tf["c"] == 0.25


def test_compute_idf():
    corpus = [
        ["a", "b"],
        ["a", "c"],
    ]
    idf = compute_idf(corpus)
    # term "a" appears in both docs → idf = log(2/2) = 0
    assert np.isclose(idf["a"], 0.0)
    # "b" appears in 1 doc → log(2/1)
    assert np.isclose(idf["b"], np.log(2))


def test_compute_tfidf_shape():
    corpus = ["a b c", "b c d"]
    tfidf_matrix, vocab = compute_tfidf(corpus)

    assert tfidf_matrix.shape == (2, len(vocab))
    assert isinstance(vocab, list)


def test_compute_tfidf_values_nonzero():
    corpus = ["a a b", "b c"]
    tfidf_matrix, vocab = compute_tfidf(corpus)

    # Ensure some TF-IDF values are non-zero
    assert np.any(tfidf_matrix > 0)
