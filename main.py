import os
import sys
import io

os.environ['IR_DATASETS_HOME'] = r"D:\ir_storage"
os.environ["PYTHONUTF8"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from services.data_service import get_dataset
from services.preprocessing_service import preprocess_text
from services.index_service import (
    INVERTED_INDEX_PATH,
    build_and_filter_index,
    load_index,
    clear_index_cache,
)
from services.retrieval_service import (
    compute_bm25_scores,
    compute_tfidf_scores,
    hybrid_parallel,
    train_word2vec,
)
from services.database_service import (
    init_db,
    get_document_count,
)
from evaluation.evaluate_system import run_evaluation_suite


# ─────────────────────────────────────────
# STEP 1 — بناء الفهرس (مرة وحدة فقط)
# ─────────────────────────────────────────
def run_indexing():
    if os.path.exists(INVERTED_INDEX_PATH):
        print("[STEP 1] Index already exists — skipping build.")
        return

    print("[STEP 1] Building inverted index on full dataset (~190K docs)...")
    print("         هاد بياخذ وقت (5-15 دقيقة) — اصبري ✓")
    ds = get_dataset()
    inv, lengths, count = build_and_filter_index(ds, min_df=2, max_df_ratio=0.9)
    print(f"[STEP 1] Done. {count} docs indexed, {len(inv)} unique terms.")


# ─────────────────────────────────────────
# STEP 2 — تدريب Word2Vec (مرة وحدة فقط)
# ─────────────────────────────────────────
def run_word2vec_training():
    if os.path.exists("data/word2vec.model"):
        print("[STEP 2] Word2Vec model already exists — skipping training.")
        return

    print("[STEP 2] Training Word2Vec on full dataset...")
    ds = get_dataset()
    train_word2vec(ds)
    print("[STEP 2] Word2Vec training complete.")


# ─────────────────────────────────────────
# STEP 3 — اختبار سريع (بدون تقييم كامل)
# ─────────────────────────────────────────
def run_quick_test():
    print("\n[STEP 3] Quick sanity check...")
    query = "lung cancer EGFR mutation treatment"

    inverted_index, doc_lengths, doc_count = load_index()
    print(f"         Index: {doc_count} docs")

    # BM25
    bm25_results = compute_bm25_scores(query)[:5]
    print(f"\n  BM25 top-5 for: '{query}'")
    for doc_id, score in bm25_results:
        print(f"    {doc_id} | {score:.4f}")

    # TF-IDF
    tfidf_results = compute_tfidf_scores(query)[:5]
    print(f"\n  TF-IDF top-5 for: '{query}'")
    for doc_id, score in tfidf_results:
        print(f"    {doc_id} | {score:.4f}")

    print("\n[STEP 3] Sanity check done.")


# ─────────────────────────────────────────
# STEP 4 — التقييم الكامل
# ─────────────────────────────────────────
def run_full_evaluation():
    print("\n[STEP 4] Running full evaluation suite...")
    print("         هاد بياخذ وقت — كل موديل بيشتغل على كل الـ queries\n")

    models = ["bm25", "tfidf", "hybrid_parallel"]    # أضيفي "bert" و "hybrid_parallel" بعد ما تتأكدي BM25 شغال صح

    all_results = {}
    for model in models:
        print(f"\n  ── Evaluating: {model} ──")
        scores = run_evaluation_suite(target_model=model)
        all_results[model] = scores

    print("\n╔══════════════════════════════════════════════════╗")
    print("║           FINAL EVALUATION SUMMARY              ║")
    print("╠══════════════════════════════════════════════════╣")
    for model, scores in all_results.items():
        print(f"║  {model.upper():<15}", end="")
        for metric, val in scores.items():
            print(f"  {metric}: {val:.4f}", end="")
        print("  ║")
    print("╚══════════════════════════════════════════════════╝")

def populate_database():
    from services.database_service import init_db
    import sqlite3
    from services.data_service import get_dataset

    # 1. التأكد من إنشاء ملف قاعدة البيانات والجدول (documents) أولاً
    init_db() 
    
    conn = sqlite3.connect("data/clinical_trials.db")
    cursor = conn.cursor()
    
    # 2. إنشاء الجدول احتياطياً في حال لم تقم الدالة السابقة بإنشائه فوراً
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            text TEXT
        )
    """)
    conn.commit()
    
    # 3. فحص إذا كانت البيانات موجودة مسبقاً لمنع التكرار
    cursor.execute("SELECT COUNT(*) FROM documents")
    if cursor.fetchone()[0] > 0:
        print("[DB] قاعدة البيانات ممتلئة وجاهزة مسبقاً ✓")
        conn.close()
        return

    print("[DB] جاري بدء تعبئة قاعدة البيانات من الـ Dataset الكاملة...")
    ds = get_dataset()
    
    batch = []
    batch_size = 5000 # حفظ كل 5000 وثيقة معاً لسرعة فائقة

    for i, doc in enumerate(ds.docs_iter()):
        title = getattr(doc, "title", "") or ""
        condition = getattr(doc, "condition", "") or ""
        summary = getattr(doc, "summary", "") or ""
        detailed_description = getattr(doc, "detailed_description", "") or ""
        eligibility = getattr(doc, "eligibility", "") or ""
        
        full_text = " ".join([title, condition, summary, detailed_description, eligibility])
        
        batch.append((doc.doc_id, full_text))
        
        if len(batch) >= batch_size:
            cursor.executemany("INSERT OR IGNORE INTO documents (doc_id, text) VALUES (?, ?)", batch)
            conn.commit()
            print(f"[⚙️ تعبئة الـ DB] تم إدخال {i+1} وثيقة بنجاح...")
            batch = []
            
    if batch:
        cursor.executemany("INSERT OR IGNORE INTO documents (doc_id, text) VALUES (?, ?)", batch)
        conn.commit()
        
    print(f"[DB] اكتملت العملية بنجاح! تم حفظ {i+1} وثيقة في قاعدة البيانات.")
    conn.close()
# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Information Retrieval System — clinicaltrials 2019")
    print("=" * 55)

    # تأكد من وجود DB
    init_db()
    populate_database()
    # الخطوات بالترتيب
   # run_indexing()           # Step 1: بناء الفهرس (مرة وحدة)
    run_word2vec_training()  
    run_quick_test()         
    run_full_evaluation()
   # clear_index_cache()
    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()