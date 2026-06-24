import os
os.environ['IR_DATASETS_HOME'] = r"D:\ir_storage"

import ir_measures
import pandas as pd
from services.data_service import get_dataset
from services.index_service import load_index
from services.retrieval_service import (
    compute_bm25_scores, 
    compute_tfidf_scores, 
    hybrid_parallel, 
    hybrid_serial
)


def run_evaluation_suite(target_model="bm25", doc_texts: dict = None):
    """
    Run evaluation suite on the official dataset and calculate aggregate metrics.
    Optimized to expand document samples and clear memory footprint.
    """
    ds = get_dataset()
    
    # 1. Parse Ground Truth (Qrels)
    qrels_list = []
    for qrel in ds.qrels_iter():
        qrels_list.append({
            'query_id': str(qrel.query_id),
            'doc_id': str(qrel.doc_id),
            'relevance': int(qrel.relevance)
        })
    qrels_df = pd.DataFrame(qrels_list)
    
    inverted_index, doc_lengths, doc_count = load_index()
    
    # 2. Dynamic Document Sampling Optimization
    # Increased sample size to 10,000 documents to boost Recall and nDCG metrics
    if doc_texts is None and target_model in ["hybrid", "hybrid_parallel", "hybrid_serial"]:
        doc_texts = {
            str(doc.doc_id): doc.text 
            for i, doc in enumerate(ds.docs_iter()) 
            if i < 10000
        }
    
    # Optimized weight distribution favoring high-performing components (BM25 & BERT)
    optimized_weights = {
        'tfidf': 0.10,
        'bm25': 0.40,
        'bert': 0.40,
        'word2vec': 0.10
    }
    
    run = []
    print(f"[INFO] Starting evaluation suite using model: {target_model}")
    
    # 3. Evaluate Queries
    for i, query_obj in enumerate(ds.queries_iter()):
        if i >= 50: 
            break
            
        q_id = str(query_obj.query_id)
        q_text = query_obj.text
        
        if target_model == "bm25":
            results = compute_bm25_scores(q_text, inverted_index, doc_lengths, doc_count)[:10]
        elif target_model == "tfidf":
            results = compute_tfidf_scores(q_text, inverted_index, doc_lengths, doc_count)[:10]
        elif target_model in ["hybrid", "hybrid_parallel"]:
            results = hybrid_parallel(
                q_text, 
                doc_texts=doc_texts, 
                top_k=10, 
                fusion_method="weighted_sum", 
                weights=optimized_weights
            )
        elif target_model == "hybrid_serial":
            results = hybrid_serial(q_text, doc_texts=doc_texts, top_k=10)
        else:
            results = []
            
        for doc_id, score in results:
            run.append({
                'query_id': q_id,
                'doc_id': str(doc_id),
                'score': float(score)
            })
            
    if not run:
        print("[WARNING] Evaluation run array is empty. Returning zero metrics.")
        return {"MAP@10": 0.0, "Recall@10": 0.0, "P@10": 0.0, "nDCG@10": 0.0}
        
    run_df = pd.DataFrame(run)
    
    # 4. Calculate Final Aggregate Evaluation Metrics
    metrics = [ir_measures.MAP@10, ir_measures.Recall@10, ir_measures.P@10, ir_measures.nDCG@10]
    results_calculated = ir_measures.calc_aggregate(metrics, qrels_df, run_df)
    
    return {str(k): float(v) for k, v in results_calculated.items()}