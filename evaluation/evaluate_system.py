import os
import pandas as pd
import ir_measures

from services.data_service import get_dataset
from services.index_service import load_index
from services.database_service import get_all_documents
from services.retrieval_service import (
    compute_bm25_scores,
    compute_tfidf_scores,
    hybrid_parallel,
    hybrid_serial,
    hybrid_clustered

)

os.environ['IR_DATASETS_HOME'] = r"D:\ir_storage"

# =========================
# GLOBAL CACHE
# =========================
# cache key = (model_name, query_text)
_QUERY_CACHE = {}


def build_query_text(query):
    """Fast safe query builder"""
    return " ".join(filter(None, [
        getattr(query, "disease", ""),
        getattr(query, "gene", ""),
        getattr(query, "demographic", "")
    ])).strip()


def run_evaluation_suite(target_model="bm25"):

    ds = get_dataset()

    # =========================
    # QRELS
    # =========================
    qrels_df = pd.DataFrame([
        {
            "query_id": str(q.query_id),
            "doc_id": str(q.doc_id),
            "relevance": int(q.relevance)
        }
        for q in ds.qrels_iter()
    ])

    # =========================
    # LOAD INDEX (cached)
    # =========================
    inverted_index, doc_lengths, doc_count = load_index()

    print(f"[INFO] docs: {len(doc_lengths)}")
    print(f"[INFO] evaluating model: {target_model}")
    doc_texts = get_all_documents()
    # =========================
    # HYBRID WEIGHTS
    # =========================
    optimized_weights = {
        "tfidf": 0.10,
        "bm25": 0.40,
        "bert": 0.40,
        "word2vec": 0.10
    }

    run = []

    # =========================
    # MAIN LOOP
    # =========================
    for query in ds.queries_iter():

        q_id = str(query.query_id)
        q_text = build_query_text(query)

        if not q_text:
            continue

        # ==================================
        # IMPORTANT FIX:
        # Cache depends on model + query
        # ==================================
        cache_key = (target_model, q_text)

        if cache_key in _QUERY_CACHE:
            results = _QUERY_CACHE[cache_key]

        else:

            try:

                # ---------------- BM25 ----------------
                if target_model == "bm25":

                    results = compute_bm25_scores(
                        q_text,
                        inverted_index,
                        doc_lengths,
                        doc_count
                    )[:10]

                # ---------------- TFIDF ----------------
                elif target_model == "tfidf":

                    results = compute_tfidf_scores(
                        q_text,
                        inverted_index,
                        doc_lengths,
                        doc_count
                    )[:10]

                # ---------------- HYBRID PARALLEL ----------------
                elif target_model in ["hybrid", "hybrid_parallel"]:

                    results = hybrid_parallel(
                        q_text,
                        doc_texts=None,
                        top_k=10,
                        fusion_method="weighted_sum",
                        weights=optimized_weights
                    )

                # ---------------- HYBRID SERIAL ----------------
                elif target_model == "hybrid_serial":

                    results = hybrid_serial(
                        q_text,
                        doc_texts=None,
                        top_k=10
                    )
                elif target_model == "hybrid_clustered":

                    results = hybrid_clustered(
                        q_text,
                        doc_texts,
                       top_k=10
                    )
                else:
                    results = []

                # Save to cache
                _QUERY_CACHE[cache_key] = results

            except Exception as e:
                print(f"[ERROR] Query {q_id}: {e}")
                continue

        # =========================
        # STORE RESULTS
        # =========================
        for doc_id, score in results:

            run.append({
                "query_id": q_id,
                "doc_id": str(doc_id),
                "score": float(score)
            })

    # =========================
    # EDGE CASE
    # =========================
    if not run:
        return {
            "MAP@10": 0.0,
            "Recall@10": 0.0,
            "P@10": 0.0,
            "nDCG@10": 0.0
        }

    run_df = pd.DataFrame(run)

    # =========================
    # METRICS
    # =========================
    metrics = [
        ir_measures.MAP@10,
        ir_measures.Recall@10,
        ir_measures.P@10,
        ir_measures.nDCG@10
    ]

    results = ir_measures.calc_aggregate(
        metrics,
        qrels_df,
        run_df
    )

    final_results = {
        str(k): float(v)
        for k, v in results.items()
    }

    print("\n===== RESULTS =====")

    for k, v in final_results.items():
        print(f"{k}: {v:.4f}")

    return final_results

