import os
import sys
import io

os.environ['IR_DATASETS_HOME'] = r"D:\ir_storage"
os.environ["PYTHONUTF8"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from services.data_service import get_dataset, count_documents, SUPPORTED_DATASETS
from services.preprocessing_service import preprocess_text
from services.index_service import (
    build_and_filter_index, 
    load_index, 
    get_index_stats, 
    clear_index_cache
)
from services.retrieval_service import (
    compute_tfidf_scores,
    compute_bm25_scores,
    compute_embedding_scores,
    hybrid_serial,
    hybrid_parallel,
    train_word2vec
)
from services.database_service import get_document_count, init_db, get_documents_by_ids

os.environ["PYTHONUTF8"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def run_data_validation():
    try:
        ds = get_dataset()
        count = count_documents(ds)
        n_q = sum(1 for _ in ds.queries_iter())
        n_qrels = sum(1 for _ in ds.qrels_iter())
        
        print(f"Docs count: {count:,}")
        print(f"Queries count: {n_q:,}")
        print(f"Qrels count: {n_qrels:,}")
        print(f"Requirement >= 200K docs: {count >= 200000}")
    except Exception as e:
        print(f"Error validating data: {e}")


def run_preprocessing_test():
    try:
        for name, ds_id in SUPPORTED_DATASETS.items():
            print(f"Checking dataset: {name}")
            ds = get_dataset(ds_id)
            
            for i, doc in enumerate(ds.docs_iter()):
                if i >= 2:
                    break
                result = preprocess_text(doc.text)
                print(f"Doc ID: {doc.doc_id}")
                print(f"  Original: {doc.text[:90]}...")
                print(f"  Processed: {result['final_text'][:90]}...")
    except Exception as e:
        print(f"Error testing preprocessing: {e}")


def run_indexing():
    print("Building inverted index and loading documents... Please wait...")
    ds = get_dataset()
    inverted_index, doc_lengths, doc_count = build_and_filter_index(ds, min_df=2, max_df_ratio=0.9)
    print("Indexing completed successfully!")
    stats = get_index_stats(inverted_index, doc_lengths, doc_count)
    print(f"Index stats: {stats}")


def run_retrieval_test(doc_texts):
    inverted_index, doc_lengths, doc_count = load_index()
    query = "atomic bomb manhattan project"
    
    print("\n--- TF-IDF Retrieval ---")
    tfidf_results = compute_tfidf_scores(query, inverted_index, doc_lengths, doc_count)
    for doc_id, score in tfidf_results[:5]:
        print(f"Doc ID: {doc_id} | Score: {score:.4f}")
    
    print("\n--- BM25 Retrieval ---")
    bm25_results = compute_bm25_scores(query, inverted_index, doc_lengths, doc_count, k1=1.5, b=0.75)
    for doc_id, score in bm25_results[:5]:
        print(f"Doc ID: {doc_id} | Score: {score:.4f}")


def run_embedding_test(doc_texts):
    query = "atomic bomb manhattan project"
    print("\n--- Dense Embedding Retrieval ---")
    results = compute_embedding_scores(query, doc_texts, top_k=5)
    for doc_id, score in results:
        print(f"Doc ID: {doc_id} | Score: {score:.4f}")


def run_hybrid_test(doc_texts):
    query = "atomic bomb manhattan project"
    
    print("\n--- Hybrid Serial (Reranking) ---")
    serial_results = hybrid_serial(query, doc_texts, top_k=5)
    for doc_id, score in serial_results:
        print(f"Doc ID: {doc_id} | Score: {score:.4f}")

    print("\n--- Hybrid Parallel (RRF) ---")
    parallel_rrf = hybrid_parallel(query, doc_texts, top_k=5, fusion_method="rrf")
    for doc_id, score in parallel_rrf:
        print(f"Doc ID: {doc_id} | Score: {score:.4f}")

    print("\n--- Hybrid Parallel (Weighted Sum) ---")
    parallel_ws = hybrid_parallel(query, doc_texts, top_k=5, fusion_method="weighted_sum")
    for doc_id, score in parallel_ws:
        print(f"Doc ID: {doc_id} | Score: {score:.4f}")


def run_database_test():
    db_count = get_document_count()
    print(f"Total documents stored in Database: {db_count:,}")
    
    inverted_index, doc_lengths, doc_count = load_index()
    query = "atomic bomb manhattan project"
    
    bm25_results = compute_bm25_scores(query, inverted_index, doc_lengths, doc_count, k1=1.5, b=0.75)
    top_10_ids = [doc_id for doc_id, _ in bm25_results[:10]]
    
    db_documents = get_documents_by_ids(top_10_ids)
    
    print("\n--- Database Content Retrieval (Top 3) ---")
    for doc_id in top_10_ids[:3]:
        raw_text = db_documents.get(doc_id, "Missing Text")
        print(f"Doc ID: {doc_id} | Original Text Preview: {raw_text[:120]}...")


def main():
    print("Initializing System Pipeline...")
    
    # 1. Indexing & Offline Storage
    run_indexing()

    ds = get_dataset()
    train_word2vec(ds, max_docs=1000)

    # 2. Prepare Sample Texts from Cache for Dense/Hybrid Models
    doc_texts = {}
    for i, doc in enumerate(ds.docs_iter()):
        if i >= 100:
            break
        doc_texts[doc.doc_id] = doc.text

    # 3. Execution of Pipeline Tests
    run_data_validation()
    run_preprocessing_test()
    run_retrieval_test(doc_texts)
    run_embedding_test(doc_texts)
    run_hybrid_test(doc_texts)
    run_database_test()
    
    clear_index_cache()
    print("Pipeline Execution Completed.")


if __name__ == "__main__":
    main()