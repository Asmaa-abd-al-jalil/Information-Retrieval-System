import os
os.environ["PYTHONUTF8"] = "1"

from services.preprocessing_service import preprocess_text
from services.retrieval_service import (
    compute_tfidf_scores, compute_bm25_scores,
    compute_embedding_scores, compute_word2vec_scores,
    hybrid_serial, hybrid_parallel
)
from services.index_service import load_index

def process_query(query: str) -> dict:
    """
    معالجة الاستعلام بنفس طريقة معالجة الوثائق
    """
    result = preprocess_text(query)
    return {
        "original":     query,
        "normalized":   result['normalized'],
        "tokens":       result['tokens'],
        "tokens_no_stop": result['tokens_no_stop'],
        "final_tokens": result['final_tokens'],
        "final_text":   result['final_text']
    }

from query.query_refinement_service import correct_query
def search(query: str, model: str = "tfidf", top_k: int = 10,
           k1: float = 1.5, b: float = 0.75,
           doc_texts: dict = None,
           fusion_method: str = "rrf",
           weights: dict = None) -> dict:
    refined_query = correct_query(query)
    if refined_query != query:
        print(f" التصحيح الإملائي: '{query}' -> '{refined_query}'")
    # معالجة الاستعلام
    processed_query = process_query(refined_query)
    print(f"🔍 الاستعلام الأصلي  : {processed_query['original']}")
    print(f"🔍 الاستعلام المعالج : {processed_query['final_text']}")

    # ← التعديل: BERT و Word2Vec يأخذوا النص الأصلي
    if model in ["bert", "word2vec", "hybrid_serial", "hybrid_parallel"]:
        query_text = processed_query['original']
    else:
        query_text = processed_query['final_text']

    inverted_index, doc_lengths, doc_count = load_index()

    if model == "tfidf":
        results = compute_tfidf_scores(
            query_text, inverted_index, doc_lengths, doc_count
        )[:top_k]

    elif model == "bm25":
        results = compute_bm25_scores(
            query_text, inverted_index, doc_lengths, doc_count,
            k1=k1, b=b
        )[:top_k]

    elif model == "bert":
        if not doc_texts:
            raise ValueError("❌ لازم تعطي doc_texts عند استخدام bert")
        results = compute_embedding_scores(
            query_text, doc_texts, top_k=top_k
        )

    elif model == "word2vec":
        if not doc_texts:
            raise ValueError("❌ لازم تعطي doc_texts عند استخدام word2vec")
        results = compute_word2vec_scores(
            query_text, doc_texts, top_k=top_k
        )

    elif model == "hybrid_serial":
     if not doc_texts:
        raise ValueError("❌ لازم تعطي doc_texts عند استخدام hybrid_serial")
     results = hybrid_serial(
        query_text,
        doc_texts,
        top_k=top_k,
        k1=k1, b=b,
        original_query=processed_query['original']
    )

    elif model == "hybrid_parallel":
        if not doc_texts:
            raise ValueError("❌ لازم تعطي doc_texts عند استخدام hybrid_parallel")
        results = hybrid_parallel(
            query_text, doc_texts, top_k=top_k, k1=k1, b=b,
            fusion_method=fusion_method, weights=weights
        )

    else:
        raise ValueError(f"❌ نموذج غير معرف: {model}")

    return {
        "query":        processed_query,
        "model":        model,
        "top_k":        top_k,
        "results":      results,
        "result_count": len(results)
    }