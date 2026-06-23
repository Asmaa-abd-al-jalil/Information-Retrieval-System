import ir_measures
import pandas as pd
from services.retrieval_service import retrieve

def run_evaluation_suite(target_model="bm25"):
    QUERIES_PATH = r"D:\ir_storage\msmarco-passage\trec-dl-2019\msmarco-test2019-queries.tsv"
    QRELS_PATH = r"D:\ir_storage\msmarco-passage\trec-dl-2019\qrels" 
    
    queries = pd.read_csv(QUERIES_PATH, sep='\t', names=['query_id', 'text'])
    queries['query_id'] = queries['query_id'].astype(str)
    
    qrels = pd.read_csv(QRELS_PATH, sep=' ', names=['query_id', 'q0', 'doc_id', 'relevance'])
    qrels['query_id'] = qrels['query_id'].astype(str)
    qrels['doc_id'] = qrels['doc_id'].astype(str)
    
    run = []
    print(f" بدء التقييم للنموذج: {target_model}")
    
    for _, row in queries.iterrows():
        q_id = row['query_id'] 
        q_text = row['text']
        
        results = retrieve(q_text, model=target_model, top_k=10)
        
        for doc_id, score in results:
            run.append({'query_id': q_id, 'doc_id': str(doc_id), 'score': float(score)})
    
    run_df = pd.DataFrame(run)
    run_df['query_id'] = run_df['query_id'].astype(str)
    run_df['doc_id'] = run_df['doc_id'].astype(str)
    
    metrics = [ir_measures.MAP@10, ir_measures.Recall@10, ir_measures.P@10, ir_measures.nDCG@10]
    
    results = ir_measures.calc_aggregate(metrics, qrels, run_df)
    
    return {str(k): float(v) for k, v in results.items()}