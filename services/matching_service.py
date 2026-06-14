from services.retrieval_service import (
    compute_tfidf_scores,
    compute_bm25_scores,
    compute_embedding_scores,
    compute_word2vec_scores
)
from services.index_service import load_index


def rank_tfidf(query, top_k=10):
    inverted_index, doc_lengths, doc_count = load_index()

    return compute_tfidf_scores(
        query,
        inverted_index,
        doc_lengths,
        doc_count
    )[:top_k]


def rank_bm25(query, top_k=10, k1=1.5, b=0.75):
    inverted_index, doc_lengths, doc_count = load_index()

    return compute_bm25_scores(
        query,
        inverted_index,
        doc_lengths,
        doc_count,
        k1,
        b
    )[:top_k]


def rank_bert(query, doc_texts, top_k=10):
    return compute_embedding_scores(
        query,
        doc_texts,
        top_k
    )


def rank_word2vec(query, doc_texts, top_k=10):
    return compute_word2vec_scores(
        query,
        doc_texts,
        top_k
    )


def rank_documents(
    query,
    model="bm25",
    top_k=10,
    doc_texts=None
):
    """
    Matching & Ranking Layer
    """

    if model == "tfidf":
        return rank_tfidf(query, top_k)

    elif model == "bm25":
        return rank_bm25(query, top_k)

    elif model == "bert":
        return rank_bert(query, doc_texts, top_k)

    elif model == "word2vec":
        return rank_word2vec(query, doc_texts, top_k)

    else:
        raise ValueError(f"Unknown model: {model}")