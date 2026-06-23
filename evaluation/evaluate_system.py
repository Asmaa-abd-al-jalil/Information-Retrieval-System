import ir_measures
import pandas as pd
from services.data_service import get_dataset
from services.index_service import load_index
from services.retrieval_service import compute_bm25_scores, compute_tfidf_scores

def run_evaluation_suite(target_model="bm25"):
    ds = get_dataset()
    
    qrels_list = []
    for qrel in ds.qrels_iter():
        qrels_list.append({
            'query_id': str(qrel.query_id),
            'doc_id': str(qrel.doc_id),
            'relevance': int(qrel.relevance)
        })
    qrels_df = pd.DataFrame(qrels_list)
    
    inverted_index, doc_lengths, doc_count = load_index()
    
    run = []
    print(f"Starting evaluation suite for official dataset using model: {target_model}")
    
    for i, query_obj in enumerate(ds.queries_iter()):
        if i >= 50: 
            break
            
        q_id = str(query_obj.query_id)
        q_text = query_obj.text
        
        if target_model == "bm25":
            results = compute_bm25_scores(q_text, inverted_index, doc_lengths, doc_count)[:10]
        elif target_model == "tfidf":
            results = compute_tfidf_scores(q_text, inverted_index, doc_lengths, doc_count)[:10]
        else:
            results = []
            
        for doc_id, score in results:
            run.append({
                'query_id': q_id,
                'doc_id': str(doc_id),
                'score': float(score)
            })
            
    if not run:
        return {"MAP@10": 0.0, "Recall@10": 0.0, "P@10": 0.0, "nDCG@10": 0.0}
        
    run_df = pd.DataFrame(run)
    
    metrics = [ir_measures.MAP@10, ir_measures.Recall@10, ir_measures.P@10, ir_measures.nDCG@10]
    results_calculated = ir_measures.calc_aggregate(metrics, qrels_df, run_df)
    
    return {str(k): float(v) for k, v in results_calculated.items()}