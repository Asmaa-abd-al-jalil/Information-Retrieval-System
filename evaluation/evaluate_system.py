import os
import pandas as pd
import ir_measures
from ir_measures import ScoredDoc, Qrel

from services.data_service import get_dataset
from services.index_service import load_index
from services.database_service import get_all_documents

from services.retrieval_service import (
    compute_bm25_scores,
    compute_tfidf_scores,
    hybrid_parallel,
    hybrid_serial,
    compute_embedding_scores,
)

os.environ['IR_DATASETS_HOME'] = r"D:\ir_storage"

# =========================
# TOP-K FOR EVALUATION
# لازم top-1000 عشان MAP يحسب صح
# top-10 بيخلي MAP ≈ 1% حتى لو النظام ممتاز
# =========================
EVAL_TOP_K = 1000

# =========================
# QUERY CACHE
# =========================
_QUERY_CACHE = {}


def build_query_text(query) -> str:
    """
    trec-pm-2019 queries فيها: disease, gene, demographic
    بنجمعهم بمسافة واحدة
    """
    parts = [
        getattr(query, "disease", "") or "",
        getattr(query, "gene", "") or "",
        getattr(query, "demographic", "") or "",
    ]
    return " ".join(p.strip() for p in parts if p.strip())


def _load_qrels(ds) -> pd.DataFrame:
    """
    ir_measures بيحتاج:
      query_id  (str)
      doc_id    (str)
      relevance (int)
    """
    rows = []
    for q in ds.qrels_iter():
        rows.append({
            "query_id":  str(q.query_id),
            "doc_id":    str(q.doc_id),
            "relevance": int(q.relevance),
        })
    df = pd.DataFrame(rows)
    print(f"[INFO] Loaded {len(df)} qrels for {df['query_id'].nunique()} queries")
    return df


def _load_doc_texts() -> dict:
    """
    نجيب كل الوثائق من DB مرة وحدة فقط
    """
    docs = get_all_documents()
    print(f"[INFO] Loaded {len(docs)} documents from DB")
    return docs


def run_evaluation_suite(target_model: str = "bm25") -> dict:
    """
    يحسب MAP, Recall, P@10, nDCG@10 لكل موديل.

    target_model: "bm25" | "tfidf" | "bert" | "hybrid_serial" | "hybrid_parallel"
    """

    ds = get_dataset()

    # --- qrels ---
    qrels_df = _load_qrels(ds)

    # --- index ---
    inverted_index, doc_lengths, doc_count = load_index()
    print(f"[INFO] Index loaded: {doc_count} docs, {len(inverted_index)} terms")

    # --- doc texts (للـ embedding والـ hybrid) ---
    doc_texts = None
    if target_model in ("bert", "hybrid_serial", "hybrid_parallel"):
        doc_texts = _load_doc_texts()

    # --- hybrid weights ---
    hybrid_weights = {
        "tfidf":    0.10,
        "bm25":     0.40,
        "bert":     0.40,
        "word2vec": 0.10,
    }

    # =========================
    # MAIN LOOP
    # =========================
    run_rows = []

    for query in ds.queries_iter():
        q_id   = str(query.query_id)
        q_text = build_query_text(query)

        if not q_text:
            print(f"[WARN] Empty query: {q_id}")
            continue

        # cache key = (model, query_text)
        cache_key = (target_model, q_text)

        if cache_key in _QUERY_CACHE:
            results = _QUERY_CACHE[cache_key]

        else:
            try:
                if target_model == "bm25":
                    # ← top-1000 مو top-10
                    results = compute_bm25_scores(
                        q_text, inverted_index, doc_lengths, doc_count
                    )[:EVAL_TOP_K]

                elif target_model == "tfidf":
                    results = compute_tfidf_scores(
                        q_text, inverted_index, doc_lengths, doc_count
                    )[:EVAL_TOP_K]

                elif target_model == "bert":
                    # embedding بيشتغل على كل الوثائق
                    results = compute_embedding_scores(
                        q_text, doc_texts, top_k=EVAL_TOP_K
                    )

                elif target_model == "hybrid_serial":
                    # serial: BM25 أولاً → top-50 → embedding
                    # نرفع top_k للتقييم
                    results = hybrid_serial(
                        q_text, doc_texts, top_k=EVAL_TOP_K
                    )

                elif target_model == "hybrid_parallel":
                    results = hybrid_parallel(
                        q_text,
                        doc_texts=doc_texts,
                        top_k=EVAL_TOP_K,
                        fusion_method="weighted_sum",
                        weights=hybrid_weights,
                    )

                else:
                    print(f"[WARN] Unknown model: {target_model}")
                    results = []

                _QUERY_CACHE[cache_key] = results

            except Exception as e:
                print(f"[ERROR] query {q_id}: {e}")
                continue

        # =========================
        # اجمع النتائج
        # =========================
        for rank, (doc_id, score) in enumerate(results):
            run_rows.append({
                "query_id": q_id,
                "doc_id":   str(doc_id),
                "score":    float(score),
            })

    # =========================
    # EDGE CASE
    # =========================
    if not run_rows:
        print("[ERROR] No results returned — check model and data")
        return {"MAP": 0.0, "Recall@1000": 0.0, "P@10": 0.0, "nDCG@10": 0.0}

    run_df = pd.DataFrame(run_rows)
    print(f"[INFO] Total scored pairs: {len(run_df)}")

    # =========================
    # DEBUG: تحقق من تطابق الـ IDs
    # =========================
    qrel_doc_ids = set(qrels_df["doc_id"].unique())
    run_doc_ids  = set(run_df["doc_id"].unique())
    overlap = qrel_doc_ids & run_doc_ids
    print(f"[DEBUG] qrel docs: {len(qrel_doc_ids)} | run docs: {len(run_doc_ids)} | overlap: {len(overlap)}")

    if len(overlap) == 0:
        print("[CRITICAL] Zero overlap between qrels and run — doc_id format mismatch!")
        print(f"  qrel sample: {list(qrel_doc_ids)[:3]}")
        print(f"  run  sample: {list(run_doc_ids)[:3]}")

    # =========================
    # METRICS
    # ir_measures بيحتاج:
    #   MAP بدون @K عشان يحسب على كل الـ relevant docs
    #   P@10 و nDCG@10 و Recall@1000 للتقرير
    # =========================
    metrics = [
        ir_measures.MAP,          # MAP كامل (مو MAP@10)
        ir_measures.Recall @ 1000,
        ir_measures.P @ 10,
        ir_measures.nDCG @ 10,
    ]

    try:
        aggregated = ir_measures.calc_aggregate(metrics, qrels_df, run_df)
        final = {str(k): round(float(v), 4) for k, v in aggregated.items()}
    except Exception as e:
        print(f"[ERROR] ir_measures failed: {e}")
        return {}

    print("\n========== EVALUATION RESULTS ==========")
    print(f"  Model : {target_model}")
    for k, v in final.items():
        print(f"  {k:30s}: {v:.4f}")
    print("=========================================\n")

    return final
