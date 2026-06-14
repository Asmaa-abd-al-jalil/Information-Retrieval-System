from services.matching_service import rank_documents
from services.data_service import get_dataset
from services.retrieval_service import train_word2vec

print("=== Matching & Ranking Test ===")

# تحميل البيانات
ds = get_dataset()
from services.index_service import build_and_filter_index
build_and_filter_index(ds, max_docs=1000)

# تدريب Word2Vec إذا كان مطلوب
train_word2vec(ds, max_docs=1000)

# تجهيز مجموعة وثائق صغيرة للاختبار
doc_texts = {}

for i, doc in enumerate(ds.docs_iter()):
    if i >= 100:
        break

    doc_texts[doc.doc_id] = doc.text

query = "atomic bomb manhattan project"

for model in ["tfidf", "bm25", "bert", "word2vec"]:

    print(f"\n--- {model.upper()} ---")

    results = rank_documents(
        query=query,
        model=model,
        top_k=5,
        doc_texts=doc_texts
    )

    for doc_id, score in results:
        print(f"{doc_id}: {score:.4f}")