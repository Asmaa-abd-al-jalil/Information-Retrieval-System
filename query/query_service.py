import os
os.environ["PYTHONUTF8"] = "1"

from services.preprocessing_service import preprocess_text
from services.retrieval_service import (
    compute_tfidf_scores,
    compute_bm25_scores,
    compute_embedding_scores,
    compute_word2vec_scores,
    hybrid_serial,
    hybrid_parallel,
)
from services.index_service import load_index
from services.database_service import get_all_documents

# query refinement اختياري — لا يكسر الكود إذا مو موجود
try:
    from query.query_refinement_service import correct_query
    _HAS_REFINEMENT = True
except ImportError:
    _HAS_REFINEMENT = False


def process_query(query: str) -> dict:
    """
    نمرر الـ query عبر نفس الـ pipeline المستخدم في الوثائق.
    يرجع dict بكل الـ keys الصحيحة.
    """
    result = preprocess_text(query)
    # كل الـ keys موجودة من preprocessing_service المُصلَح
    return {
        "original":       result["original"],
        "normalized":     result["normalized"],
        "tokens":         result["tokens"],
        "tokens_no_stop": result["tokens_no_stop"],
        "final_tokens":   result["final_tokens"],
        "final_text":     result["final_text"],
    }


def search(
    query: str,
    model: str = "bm25",
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
    doc_texts: dict = None,
    fusion_method: str = "rrf",
    weights: dict = None,
) -> dict:
    """
    نقطة الدخول الرئيسية للبحث.

    model: "tfidf" | "bm25" | "bert" | "word2vec" | "hybrid_serial" | "hybrid_parallel"
    k1, b: معاملات BM25 (قابلة للتحكم من الـ UI)
    weights: أوزان الـ fusion للـ hybrid_parallel
    """

    # ─── 1. Query Refinement (تصحيح إملائي) ───
    refined = query
    if _HAS_REFINEMENT:
        try:
            refined = correct_query(query)
            if refined != query:
                print(f"[INFO] Spell correction: '{query}' → '{refined}'")
        except Exception as e:
            print(f"[WARN] Refinement failed: {e}")

    # ─── 2. Preprocessing ───
    processed = process_query(refined)
    print(f"[INFO] Query original : {processed['original']}")
    print(f"[INFO] Query processed: {processed['final_text']}")

    # ─── 3. Load index ───
    inverted_index, doc_lengths, doc_count = load_index()

    # ─── 4. Load doc_texts إذا محتاجين ───
    needs_texts = model in ("bert", "word2vec", "hybrid_serial", "hybrid_parallel")
    if needs_texts and doc_texts is None:
        doc_texts = get_all_documents()

    # ─── 5. Routing ───
    if model == "tfidf":
        results = compute_tfidf_scores(
            processed["final_text"],
            inverted_index, doc_lengths, doc_count,
        )[:top_k]

    elif model == "bm25":
        results = compute_bm25_scores(
            processed["final_text"],
            inverted_index, doc_lengths, doc_count,
            k1=k1, b=b,
        )[:top_k]

    elif model == "bert":
        results = compute_embedding_scores(
            processed["original"],   # Embedding أفضل مع النص الأصلي
            doc_texts,
            top_k=top_k,
        )

    elif model == "word2vec":
        results = compute_word2vec_scores(
            processed["original"],
            doc_texts,
            top_k=top_k,
        )

    elif model == "hybrid_serial":
        results = hybrid_serial(
            processed["original"],
            doc_texts,
            top_k=top_k,
        )

    elif model == "hybrid_parallel":
        results = hybrid_parallel(
            processed["original"],
            doc_texts=doc_texts,
            top_k=top_k,
            fusion_method=fusion_method,
            weights=weights,
        )

    else:
        raise ValueError(f"Unknown model: '{model}'")

    return {
        "query":        processed,
        "model":        model,
        "top_k":        top_k,
        "results":      results,
        "result_count": len(results),
    }
