import os

os.environ['IR_DATASETS_HOME'] = r"D:\ir_storage"

import ir_measures
import pandas as pd

from services.data_service import get_dataset
from services.index_service import load_index
from services.database_service import get_all_documents
from services.retrieval_service import (
    compute_bm25_scores,
    compute_tfidf_scores,
    hybrid_parallel,
    hybrid_serial
)


def run_evaluation_suite(
        target_model="bm25",
        doc_texts: dict = None
):
    """
    Evaluate retrieval models using ir_measures.

    Parameters:
        target_model : bm25 | tfidf | hybrid | hybrid_parallel | hybrid_serial
        doc_texts    : dictionary {doc_id: text}

    Returns:
        dict containing aggregate metrics.
    """

    ds = get_dataset()

    # ======================================================
    # Build Qrels DataFrame
    # ======================================================

    qrels_list = []

    for qrel in ds.qrels_iter():
        qrels_list.append({
            'query_id': str(qrel.query_id),
            'doc_id': str(qrel.doc_id),
            'relevance': int(qrel.relevance)
        })

    qrels_df = pd.DataFrame(qrels_list)

    # ======================================================
    # Load index
    # ======================================================

    inverted_index, doc_lengths, doc_count = load_index()

    # ======================================================
    # Load documents automatically for hybrid models
    # ======================================================

    if target_model in [
        "hybrid",
        "hybrid_parallel",
        "hybrid_serial"
    ]:

        if doc_texts is None:

            print("[INFO] Loading documents from database...")

            doc_texts = get_all_documents()

            print(
                f"[INFO] Loaded {len(doc_texts)} documents."
            )

    # ======================================================
    # Hybrid weights
    # ======================================================

    optimized_weights = {
        'tfidf': 0.10,
        'bm25': 0.40,
        'bert': 0.40,
        'word2vec': 0.10
    }

    run = []

    print(
        f"[INFO] Starting evaluation using: {target_model}"
    )

    # ======================================================
    # Evaluate all queries
    # ======================================================

    for query_obj in ds.queries_iter():

        q_id = str(query_obj.query_id)

        # بناء نص الاستعلام من حقول TREC PM
        q_text = " ".join(filter(None, [
            getattr(query_obj, "disease", ""),
            getattr(query_obj, "gene", ""),
            getattr(query_obj, "demographic", "")
        ])).strip()

        try:

            # --------------------------------------------------
            # BM25
            # --------------------------------------------------

            if target_model == "bm25":

                results = compute_bm25_scores(
                    q_text,
                    inverted_index,
                    doc_lengths,
                    doc_count
                )[:10]

            # --------------------------------------------------
            # TF-IDF
            # --------------------------------------------------

            elif target_model == "tfidf":

                results = compute_tfidf_scores(
                    q_text,
                    inverted_index,
                    doc_lengths,
                    doc_count
                )[:10]

            # --------------------------------------------------
            # Hybrid Parallel
            # --------------------------------------------------

            elif target_model in [
                "hybrid",
                "hybrid_parallel"
            ]:

                if not doc_texts:

                    print(
                        "[WARNING] doc_texts is missing."
                    )

                    continue

                results = hybrid_parallel(
                    q_text,
                    doc_texts=doc_texts,
                    top_k=10,
                    fusion_method="weighted_sum",
                    weights=optimized_weights
                )

            # --------------------------------------------------
            # Hybrid Serial
            # --------------------------------------------------

            elif target_model == "hybrid_serial":

                if not doc_texts:

                    print(
                        "[WARNING] doc_texts is missing."
                    )

                    continue

                results = hybrid_serial(
                    q_text,
                    doc_texts=doc_texts,
                    top_k=10
                )

            else:

                results = []

            # --------------------------------------------------
            # Save results
            # --------------------------------------------------

            for doc_id, score in results:

                run.append({
                    'query_id': q_id,
                    'doc_id': str(doc_id),
                    'score': float(score)
                })

        except Exception as e:

            print(
                f"[ERROR] Query {q_id} failed: {str(e)}"
            )

    # ======================================================
    # No results protection
    # ======================================================

    if not run:

        print(
            "[WARNING] No retrieval results generated."
        )

        return {
            "MAP@10": 0.0,
            "Recall@10": 0.0,
            "P@10": 0.0,
            "nDCG@10": 0.0
        }

    run_df = pd.DataFrame(run)

    # ======================================================
    # Evaluate only processed queries
    # ======================================================

    evaluated_queries = set(run_df["query_id"])

    filtered_qrels = qrels_df[
        qrels_df["query_id"].isin(evaluated_queries)
    ]

    # ======================================================
    # Metrics
    # ======================================================

    metrics = [
        ir_measures.MAP@10,
        ir_measures.Recall@10,
        ir_measures.P@10,
        ir_measures.nDCG@10
    ]

    results_calculated = ir_measures.calc_aggregate(
        metrics,
        filtered_qrels,
        run_df
    )

    final_results = {
        str(metric): float(score)
        for metric, score in results_calculated.items()
    }

    print("\n===== Evaluation Results =====")

    for metric, value in final_results.items():
        print(f"{metric}: {value:.4f}")

    return final_results