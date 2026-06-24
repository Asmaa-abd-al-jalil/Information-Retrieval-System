from services.retrieval_service import (
    compute_tfidf_scores,
    compute_bm25_scores,
    compute_embedding_scores,
    compute_word2vec_scores,
    hybrid_serial,
    hybrid_parallel
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
    doc_texts=None,
    fusion_method="rrf",
    weights=None
):
    """
    Matching and Ranking Layer for all supported models.
    """
    if model == "tfidf":
        return rank_tfidf(query, top_k)

    elif model == "bm25":
        return rank_bm25(query, top_k)

    elif model == "bert":
        if not doc_texts:
            raise ValueError("Error: doc_texts must be provided for BERT ranking")
        return rank_bert(query, doc_texts, top_k)

    elif model == "word2vec":
        if not doc_texts:
            raise ValueError("Error: doc_texts must be provided for Word2Vec ranking")
        return rank_word2vec(query, doc_texts, top_k)

    elif model == "hybrid_serial":
        if not doc_texts:
            raise ValueError("Error: doc_texts must be provided for Hybrid Serial ranking")
        return hybrid_serial(query, doc_texts, top_k=top_k)

    elif model == "hybrid_parallel":
        if not doc_texts:
            raise ValueError("Error: doc_texts must be provided for Hybrid Parallel ranking")
        return hybrid_parallel(
            query, doc_texts, top_k=top_k, fusion_method=fusion_method, weights=weights
        )

    else:
        raise ValueError(f"Error: Unknown model: {model}")