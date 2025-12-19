import numpy as np
from collections import Counter

def tokenize(document):
    """
    Simple whitespace tokenizer.
    Pure functional: returns a new list.
    """
    return document.lower().split()


def compute_tf(doc_tokens):
    """
    Compute term frequency for a single document.
    Returns a dict: term -> tf value.
    """
    counts = Counter(doc_tokens)
    total = len(doc_tokens)
    return {term: count / total for term, count in counts.items()}


def compute_idf(corpus_tokens):
    """
    Compute inverse document frequency across the corpus.
    corpus_tokens: list of token lists.
    Returns dict: term -> idf value.
    """
    num_docs = len(corpus_tokens)
    doc_freq = Counter()

    for tokens in corpus_tokens:
        unique_terms = set(tokens)
        for term in unique_terms:
            doc_freq[term] += 1

    return {term: np.log(num_docs / df) for term, df in doc_freq.items()}


def compute_tfidf(corpus):
    """
    Compute TF-IDF vectors for a list of documents.
    Returns:
        tfidf_matrix (numpy array)
        vocabulary (list of terms in consistent order)
    """
    corpus_tokens = [tokenize(doc) for doc in corpus]

    tf_list = [compute_tf(tokens) for tokens in corpus_tokens]
    idf = compute_idf(corpus_tokens)

    vocabulary = sorted(idf.keys())

    tfidf_matrix = np.zeros((len(corpus), len(vocabulary)))

    for i, tf in enumerate(tf_list):
        for j, term in enumerate(vocabulary):
            tfidf_matrix[i, j] = tf.get(term, 0.0) * idf[term]

    return tfidf_matrix, vocabulary
