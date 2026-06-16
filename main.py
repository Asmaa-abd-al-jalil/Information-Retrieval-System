import os
os.environ["PYTHONUTF8"] = "1"  # ← لازم يكون قبل أي import آخر
import sys
import io
from services.data_service import get_dataset, count_documents, get_processed_doc_text
from services.preprocessing_service import preprocess_text
from services.index_service import build_inverted_index, save_index, get_index_stats
from services.retrieval_service import retrieve

# ضبط الترميز للنظام
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_data_validation():
    """المرحلة الأولى: التحقق من البيانات"""
    print("\n🚀 المرحلة الأولى: التحقق من البيانات")
    print("-" * 50)
    try:
        ds = get_dataset()
        count = count_documents(ds)
        n_q = sum(1 for _ in ds.queries_iter())
        n_qrels = sum(1 for _ in ds.qrels_iter())
        
        print(f"  ✅ الوثائق  : {count:,}")
        print(f"  ✅ Queries  : {n_q:,}")
        print(f"  ✅ Qrels    : {n_qrels:,}")
        print(f"  ✅ شرط 200K : {'محقق ✅' if count > 200_000 else 'غير محقق ❌'}")
    except Exception as e:
        print(f"  ❌ خطأ أثناء التحميل: {e}")

def run_preprocessing_test():
    """المرحلة الثانية: اختبار المعالجة الأولية"""
    print("\n🚀 المرحلة الثانية: اختبار المعالجة (Preprocessing Test)")
    print("-" * 50)
    try:
        ds = get_dataset()
        sample_doc = next(iter(ds.docs_iter()))
        
        print("--- النص الأصلي ---")
        print(sample_doc.text[:200])
        
        print("\n--- النص بعد المعالجة ---")
        processed = get_processed_doc_text(sample_doc)
        print(processed[:200])
    except Exception as e:
        print(f"  ❌ خطأ أثناء المعالجة: {e}")

def run_preprocessing_test():
    print("\n🚀 اختبار Preprocessing على داتا حقيقية")
    print("-" * 50)
    
    from services.data_service import get_dataset, SUPPORTED_DATASETS
    from services.preprocessing_service import preprocess_text
    
    for name, ds_id in SUPPORTED_DATASETS.items():
        print(f"\n📌 Dataset: {name}")
        ds = get_dataset(ds_id)
        
        # خذ 3 وثائق عشوائية
        for i, doc in enumerate(ds.docs_iter()):
            if i >= 3:
                break
            result = preprocess_text(doc.text)
            print(f"  Original : {doc.text[:100]}")
            print(f"  Processed: {result['final_text'][:100]}")
            print()  
            
def run_indexing():
    print("\n🚀 المرحلة الثالثة: بناء الـ Index")
    print("-" * 50)
    from services.index_service import build_and_filter_index, get_index_stats
    from services.data_service import get_dataset
    
    ds = get_dataset()
    inverted_index, doc_lengths, doc_count = build_and_filter_index(
        ds, 
        max_docs=1000,
        min_df=2,
        max_df_ratio=0.9
    )
    
    stats = get_index_stats(inverted_index, doc_lengths, doc_count)
    print(f"📊 إحصائيات: {stats}")

def run_retrieval_test():
    print("\n🚀 اختبار الاسترجاع")
    print("-" * 50)
    
    query = "atomic bomb manhattan project"
    
    print(f"🔍 الاستعلام: {query}")
    
    print("\n--- TF-IDF ---")
    results = retrieve(query, model="tfidf", top_k=5)
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.4f}")
    
    print("\n--- BM25 ---")
    results = retrieve(query, model="bm25", top_k=5, k1=1.5, b=0.75)
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.4f}") 

def run_embedding_test():
    print("\n🚀 اختبار Embedding")
    print("-" * 50)
    
    from services.data_service import get_dataset
    from services.retrieval_service import retrieve

    query = "atomic bomb manhattan project"
    
    # خذ 100 وثيقة للاختبار
    ds = get_dataset()
    doc_texts = {}
    for i, doc in enumerate(ds.docs_iter()):
        if i >= 100:
            break
        doc_texts[doc.doc_id] = doc.text

    print(f"🔍 الاستعلام: {query}")
    results = retrieve(query, model="embedding", top_k=5, doc_texts=doc_texts)
    
    print("\n--- Embedding Results ---")
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.4f}")

def run_hybrid_test():
    print("\n🚀 اختبار Hybrid")
    print("-" * 50)
    
    from services.data_service import get_dataset
    from services.retrieval_service import hybrid_serial, hybrid_parallel, train_word2vec

    query = "atomic bomb manhattan project"
    
    ds = get_dataset()
    doc_texts = {}
    for i, doc in enumerate(ds.docs_iter()):
        if i >= 100:
            break
        doc_texts[doc.doc_id] = doc.text

    # تدريب Word2Vec أولاً
    train_word2vec(ds, max_docs=1000)

    # Serial
    print("\n--- Hybrid Serial ---")
    results = hybrid_serial(query, doc_texts, top_k=5)
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.4f}")

    # Parallel RRF
    print("\n--- Hybrid Parallel (RRF) ---")
    results = hybrid_parallel(query, doc_texts, top_k=5, fusion_method="rrf")
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.4f}")

    # Parallel Weighted Sum
    print("\n--- Hybrid Parallel (Weighted Sum) ---")
    results = hybrid_parallel(query, doc_texts, top_k=5, fusion_method="weighted_sum")
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.4f}")

def run_query_test():
    print("\n🚀 اختبار Query Processing")
    print("-" * 50)
    
    from services.query_service import search
    from services.data_service import get_dataset
    from services.retrieval_service import train_word2vec  # ← أضف هاد

    ds = get_dataset()
    
    # تدريب Word2Vec أولاً
    train_word2vec(ds, max_docs=1000)  # ← أضف هاد
    
    doc_texts = {}
    for i, doc in enumerate(ds.docs_iter()):
        if i >= 100:
            break
        doc_texts[doc.doc_id] = doc.text

    query = "atomic bomb manhattan project"

    for model in ["tfidf", "bm25", "bert", "word2vec"]:
        print(f"\n--- {model.upper()} ---")
        response = search(query, model=model, top_k=3, doc_texts=doc_texts)
        for doc_id, score in response['results']:
            print(f"  {doc_id}: {score:.4f}")

def run_database_test():
    print("\n🚀 اختبار الـ Database مع Ranking")
    print("-" * 50)
    
    # 1. استيراد دالة الترتيب وتحميل الفهرس
    from services.database_service import get_document_count
    from services.retrieval_service import retrieve_with_text
    from services.rank_service import rank_documents
    from services.index_service import load_index
    
    print(f"📊 عدد الوثائق بالـ Database: {get_document_count():,}")
    
    query = "atomic bomb manhattan project"
    # استرجاع مبدئي (Candidates)
    results = retrieve_with_text(query, model="tfidf", top_k=10) 
    
    # 2. تحميل الفهرس (للحصول على doc_lengths و doc_count اللازمين للترتيب)
    index, doc_lengths, doc_count = load_index()
    
    # 3. تجهيز البيانات لدالة الترتيب
    retrieved_doc_ids = [r['doc_id'] for r in results]
    query_tokens = query.lower().split() # توكنز بسيطة للاختبار
    
    # 4. استدعاء خدمة الترتيب (Ranking Service)
    ranked_results = rank_documents(query_tokens, {doc_id: {} for doc_id in retrieved_doc_ids}, index, doc_lengths, doc_count)
    
    # 5. طباعة النتائج المترتبة بـ Ranking
    print(f"\n🔍 نتائج البحث المرتبة (Ranking) عن: {query}")
    for doc_id, score in ranked_results[:3]: # نطبع أول 3 فقط
        print(f" 📄 Doc ID: {doc_id} | ⭐ Ranking Score: {score:.4f}")

def main():
    print("✨ بدء نظام استرجاع المعلومات - IR System")
    from services.data_service import get_dataset
    from services.retrieval_service import train_word2vec
   # تحكم بالمراحل هنا: يمكنك تعطيل أي مرحلة بوضع # قبلها
   # run_data_validation()
   #run_preprocessing_test()
   # run_preprocessing_test()
    run_indexing()

    ds = get_dataset()
    train_word2vec(ds, max_docs=1000)

   # run_retrieval_test()
   # run_embedding_test()
   # run_hybrid_test()
   # run_query_test()
    run_database_test()
    
    print("\n" + "-" * 50)
    print("🏁 انتهت جميع المراحل بنجاح.  ")

if __name__ == "__main__":
    main()


