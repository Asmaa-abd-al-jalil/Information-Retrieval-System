import os

os.environ['IR_DATASETS_HOME'] = r"D:\ir_storage"

import ir_measures
import pandas as pd

from services.data_service import get_dataset
from services.index_service import load_index
from services.database_service import get_documents_by_ids

from services.retrieval_service import (
    compute_bm25_scores,
    compute_tfidf_scores,
    hybrid_parallel,
    hybrid_serial
)


def run_evaluation_suite(target_model="bm25"):

    ds = get_dataset()

    # ======================================================
    # QRELS (TEST SET ONLY)
    # ======================================================
    qrels_df = pd.DataFrame([
        {
            "query_id": str(q.query_id),
            "doc_id": str(q.doc_id),
            "relevance": int(q.relevance)
        }
        for q in ds.qrels_iter()
    ])

    # ======================================================
    # LOAD INDEX
    # ======================================================
    inverted_index, doc_lengths, doc_count = load_index()

    indexed_docs = doc_lengths  # dict is enough (fast lookup + clean SOA)

    print(f"[INFO] Indexed docs: {len(indexed_docs)}")

    # ======================================================
    # LOAD DOC TEXTS (FAST - FROM DB ONLY)
    # ======================================================
    if target_model in ["hybrid", "hybrid_parallel", "hybrid_serial"]:

        print("[INFO] Loading documents from DB (indexed only)...")

        doc_texts = get_documents_by_ids(list(indexed_docs.keys()))

        print(f"[INFO] Loaded doc_texts: {len(doc_texts)}")

    else:
        doc_texts = None

    # ======================================================
    # WEIGHTS (clean separation)
    # ======================================================
    optimized_weights = {
        "tfidf": 0.10,
        "bm25": 0.40,
        "bert": 0.40,
        "word2vec": 0.10
    }

    run = []

    print(f"[INFO] Running evaluation: {target_model}")

    # ======================================================
    # EVALUATION LOOP (ONLY QUERIES - NO DOCS)
    # ======================================================
    for query in ds.queries_iter():

        q_id = str(query.query_id)

        q_text = " ".join(filter(None, [
            getattr(query, "disease", ""),
            getattr(query, "gene", ""),
            getattr(query, "demographic", "")
        ])).strip()

        if not q_text:
            continue

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

            # ---------------- HYBRID ----------------
            elif target_model in ["hybrid", "hybrid_parallel"]:

                results = hybrid_parallel(
                    q_text,
                    doc_texts=doc_texts,
                    top_k=10,
                    fusion_method="weighted_sum",
                    weights=optimized_weights
                )

            elif target_model == "hybrid_serial":

                results = hybrid_serial(
                    q_text,
                    doc_texts=doc_texts,
                    top_k=10
                )

            else:
                results = []

            # ==================================================
            # FILTER ONLY INDEXED DOCS (SAFE GUARANTEE)
            # ==================================================
            for doc_id, score in results:

                doc_id = str(doc_id)

                if doc_id not in indexed_docs:
                    continue

                run.append({
                    "query_id": q_id,
                    "doc_id": doc_id,
                    "score": float(score)
                })

        except Exception as e:
            print(f"[ERROR] Query {q_id}: {e}")

    # ======================================================
    # SAFETY CHECK
    # ======================================================
    if not run:
        return {
            "MAP@10": 0.0,
            "Recall@10": 0.0,
            "P@10": 0.0,
            "nDCG@10": 0.0
        }

    run_df = pd.DataFrame(run)

    filtered_qrels = qrels_df[
        qrels_df["query_id"].isin(run_df["query_id"])
    ]

    # ======================================================
    # METRICS (standard IR evaluation)
    # ======================================================
    metrics = [
        ir_measures.MAP@10,
        ir_measures.Recall@10,
        ir_measures.P@10,
        ir_measures.nDCG@10
    ]

    results = ir_measures.calc_aggregate(
        metrics,
        filtered_qrels,
        run_df
    )

    final_results = {
        str(k): float(v) for k, v in results.items()
    }

    print("\n===== RESULTS =====")
    for k, v in final_results.items():
        print(f"{k}: {v:.4f}")

    return final_results