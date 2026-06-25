import os
import sys
import io

os.environ['IR_DATASETS_HOME'] = r"D:\ir_storage"
os.environ["PYTHONUTF8"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from services.data_service import get_dataset, count_documents, SUPPORTED_DATASETS
from services.preprocessing_service import preprocess_text
from services.index_service import (
    INVERTED_INDEX_PATH,
    build_and_filter_index, 
    load_index, 
    clear_index_cache
)
from services.retrieval_service import (
    compute_tfidf_scores,
    compute_bm25_scores,
    compute_embedding_scores,
    hybrid_parallel
)
from services.database_service import get_document_count, init_db, get_documents_by_ids

def run_preprocessing_test():
    try:
        ds = get_dataset()
        for i, doc in enumerate(ds.docs_iter()):
            if i >= 2: break
            # دمج الحقول الصحيحة حسب الـ ClinicalTrialsDoc
            full_text = f"{doc.title} {doc.condition} {doc.summary} {doc.detailed_description} {doc.eligibility}"
            result = preprocess_text(full_text)
            print(f"Doc ID: {doc.doc_id}")
            print(f"  Processed: {result['final_text'][:90]}...")
    except Exception as e:
        print(f"Error testing preprocessing: {e}")

def run_indexing():
    if os.path.exists(INVERTED_INDEX_PATH):
        print("Index already exists. Loading from disk...")
    else:
        print("Building inverted index... Please wait...")
        ds = get_dataset()
        build_and_filter_index(ds, min_df=2, max_df_ratio=0.9)

def main():
    print("Initializing System Pipeline...")
    
    # 1. Indexing
    run_indexing()
    ds = get_dataset()

    # 2. Prepare Sample Texts (دمج الحقول بشكل صحيح)
    doc_texts = {}
    for i, doc in enumerate(ds.docs_iter()):
        if i >= 100: break
        # هذا التعديل يحل مشكلة AttributeError: 'ClinicalTrialsDoc' object has no attribute 'text'
        full_text = f"{doc.title} {doc.condition} {doc.summary} {doc.detailed_description} {doc.eligibility}"
        doc_texts[doc.doc_id] = full_text

    # 3. Execution
    run_preprocessing_test()
    
    inverted_index, doc_lengths, doc_count = load_index()
    query = "cancer treatment clinical trial"
    
    print("\n--- Running Hybrid Parallel (RRF) ---")
    results = hybrid_parallel(query, doc_texts, top_k=5, fusion_method="rrf")
    for doc_id, score in results:
        print(f"Doc ID: {doc_id} | Score: {score:.4f}")
    
    clear_index_cache()
    print("Pipeline Execution Completed.")

if __name__ == "__main__":
    main()