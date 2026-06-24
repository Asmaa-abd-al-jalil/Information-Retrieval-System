import os
import sys

os.environ["PYTHONUTF8"] = "1"

from services.preprocessing_service import preprocess_text
from services.retrieval_service import (
    compute_tfidf_scores, 
    compute_bm25_scores,
    compute_embedding_scores, 
    compute_word2vec_scores,
    hybrid_serial, 
    hybrid_parallel
)
from services.index_service import load_index
from query.query_refinement_service import correct_query

def process_query(query: str) -> dict:
    """
    Process the query using the same pipeline as documents.
    """
    result = preprocess_text(query)
    return {
        "original":         query,
        "normalized":       result['normalized'],
        "tokens":           result['tokens'],
        "tokens_no_stop":   result['tokens_no_stop'],
        "final_tokens":     result['final_tokens'],
        "final_text":       result['final_text']
    }

def search(query: str, model: str = "tfidf", top_k: int = 10,
           k1: float = 1.5, b: float = 0.75,
           doc_texts: dict = None,
           fusion_method: str = "rrf",
           weights: dict = None) -> dict:
    
    # 1. Apply spell correction
    refined_query = correct_query(query)
    if refined_query != query:
        print(f"[INFO] Spell correction applied: '{query}' -> '{refined_query}'")
        
    # 2. Preprocess the refined query
    processed_query = process_query(refined_query)
    print(f"[INFO] Original Query: {processed_query['original']}")
    print(f"[INFO] Processed Query: {processed_query['final_text']}")

    inverted_index, doc_lengths, doc_count = load_index()

    # 3. Route the query text to the appropriate model based on its mathematical design
    if model == "tfidf":
        results = compute_tfidf_scores(
            processed_query['final_text'], inverted_index, doc_lengths, doc_count
        )[:top_k]

    elif model == "bm25":
        results = compute_bm25_scores(
            processed_query['final_text'], inverted_index, doc_lengths, doc_count,
            k1=k1, b=b
        )[:top_k]

    elif model == "bert":
        if not doc_texts:
            raise ValueError("Error: doc_texts must be provided when using BERT model")
        results = compute_embedding_scores(
            processed_query['original'], doc_texts, top_k=top_k
        )

    elif model == "word2vec":
        if not doc_texts:
            raise ValueError("Error: doc_texts must be provided when using Word2Vec model")
        results = compute_word2vec_scores(
            processed_query['original'], doc_texts, top_k=top_k
        )

    elif model == "hybrid_serial":
        if not doc_texts:
            raise ValueError("Error: doc_texts must be provided when using Hybrid Serial model")
        results = hybrid_serial(
            processed_query['original'],
            doc_texts,
            top_k=top_k
        )

    elif model == "hybrid_parallel":
        if not doc_texts:
            raise ValueError("Error: doc_texts must be provided when using Hybrid Parallel model")
        results = hybrid_parallel(
            processed_query['original'], 
            doc_texts, 
            top_k=top_k, 
            k1=k1, b=b,
            fusion_method=fusion_method, 
            weights=weights
        )

    else:
        raise ValueError(f"Error: Unknown model type: {model}")

    return {
        "query":        processed_query,
        "model":        model,
        "top_k":        top_k,
        "results":      results,
        "result_count": len(results)
    }