import os
import math
import numpy as np
from collections import defaultdict
from services.preprocessing_service import preprocess_text
from services.index_service import load_index
from sentence_transformers import SentenceTransformer

def compute_tfidf_scores(query: str, inverted_index: dict, doc_lengths: dict, doc_count: int) -> dict:
    """VSM TF-IDF: حساب درجات التشابه بين الاستعلام والوثائق"""
    result = preprocess_text(query)
    query_tokens = result['final_tokens']

    scores = defaultdict(float)

    for token in query_tokens:
        if token not in inverted_index:
            continue

        # حساب الـ IDF
        df = len(inverted_index[token])
        idf = math.log((doc_count + 1) / (df + 1)) + 1

        # حساب الـ TF-IDF لكل وثيقة
        for doc_id, tf in inverted_index[token].items():
            tf_norm = 1 + math.log(tf) if tf > 0 else 0
            scores[doc_id] += tf_norm * idf

    # ترتيب النتائج
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def compute_bm25_scores(query: str, inverted_index: dict, doc_lengths: dict, 
                         doc_count: int, k1: float = 1.5, b: float = 0.75) -> list:
    """BM25: حساب درجات التشابه"""
    result = preprocess_text(query)
    query_tokens = result['final_tokens']

    avg_dl = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 1
    scores = defaultdict(float)

    for token in query_tokens:
        if token not in inverted_index:
            continue

        df = len(inverted_index[token])
        idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)

        for doc_id, tf in inverted_index[token].items():
            dl = doc_lengths.get(doc_id, avg_dl)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            scores[doc_id] += idf * tf_norm

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def retrieve(query: str, model: str = "tfidf", top_k: int = 10,
             k1: float = 1.5, b: float = 0.75, doc_texts: dict = None) -> list:
    """
    model: 'tfidf' أو 'bm25' أو 'embedding'
    doc_texts: مطلوب فقط عند model='embedding' {doc_id: text}
    """
    inverted_index, doc_lengths, doc_count = load_index()

    if model == "tfidf":
        return compute_tfidf_scores(query, inverted_index, doc_lengths, doc_count)[:top_k]
    elif model == "bm25":
        return compute_bm25_scores(query, inverted_index, doc_lengths, doc_count, k1=k1, b=b)[:top_k]
    elif model == "embedding":
        if not doc_texts:
            raise ValueError("❌ لازم تعطي doc_texts عند استخدام embedding")
        return compute_embedding_scores(query, doc_texts, top_k=top_k)
    else:
        raise ValueError(f"❌ نموذج غير معرف: {model}")


# تحميل النموذج مرة وحدة بس
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("🔄 جاري تحميل نموذج BERT...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ تم تحميل النموذج")
    return _embedding_model

def cosine_similarity(vec1, vec2):
    """حساب التشابه بين متجهين"""
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm if norm > 0 else 0.0

def compute_embedding_scores(query: str, doc_texts: dict, top_k: int = 10) -> list:
    """
    Embedding: حساب التشابه بين الاستعلام والوثائق
    doc_texts: {doc_id: text}
    """
    model = get_embedding_model()

    # تمثيل الاستعلام
    query_vec = model.encode(query, convert_to_numpy=True)

    # تمثيل الوثائق ودرجات التشابه
    scores = []
    doc_ids = list(doc_texts.keys())
    doc_vecs = model.encode(list(doc_texts.values()), convert_to_numpy=True, show_progress_bar=True)

    for doc_id, doc_vec in zip(doc_ids, doc_vecs):
        score = cosine_similarity(query_vec, doc_vec)
        scores.append((doc_id, score))

    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

def hybrid_serial(query: str, doc_texts: dict, top_k: int = 10,
                  k1: float = 1.5, b: float = 0.75) -> list:
    """
    Hybrid Serial: TF-IDF → BM25 → Embedding بالتسلسل
    كل مرحلة بتضيّق النتائج للمرحلة الجاية
    """
    inverted_index, doc_lengths, doc_count = load_index()

    # المرحلة 1: TF-IDF - خذ أفضل 100
    print("  🔄 المرحلة 1: TF-IDF...")
    tfidf_results = compute_tfidf_scores(query, inverted_index, doc_lengths, doc_count)
    top_100_ids = set(doc_id for doc_id, _ in tfidf_results[:100])

    # المرحلة 2: BM25 - على نتائج TF-IDF فقط
    print("  🔄 المرحلة 2: BM25...")
    bm25_results = compute_bm25_scores(query, inverted_index, doc_lengths, doc_count, k1=k1, b=b)
    top_50 = [(doc_id, score) for doc_id, score in bm25_results if doc_id in top_100_ids][:50]
    top_50_ids = set(doc_id for doc_id, _ in top_50)

    # المرحلة 3: Embedding - على نتائج BM25 فقط
    print("  🔄 المرحلة 3: Embedding...")
    filtered_texts = {doc_id: doc_texts[doc_id] for doc_id in top_50_ids if doc_id in doc_texts}
    
    if not filtered_texts:
        return top_50[:top_k]
    
    final_results = compute_embedding_scores(query, filtered_texts, top_k=top_k)
    return final_results


def hybrid_parallel(query: str, doc_texts: dict, top_k: int = 10,
                    k1: float = 1.5, b: float = 0.75,
                    fusion_method: str = "rrf",
                    weights: dict = None) -> list:
    """
    Hybrid Parallel: TF-IDF + BM25 + Embedding بالتوازي
    fusion_method: 'rrf' أو 'weighted_sum'
    weights: {'tfidf': 0.3, 'bm25': 0.3, 'embedding': 0.4}
    """
    if weights is None:
        weights = {'tfidf': 0.3, 'bm25': 0.3, 'embedding': 0.4}

    inverted_index, doc_lengths, doc_count = load_index()

    # تشغيل النماذج بالتوازي
    print("  🔄 TF-IDF...")
    tfidf_results = compute_tfidf_scores(query, inverted_index, doc_lengths, doc_count)
    
    print("  🔄 BM25...")
    bm25_results = compute_bm25_scores(query, inverted_index, doc_lengths, doc_count, k1=k1, b=b)
    
    print("  🔄 Embedding...")
    embedding_results = compute_embedding_scores(query, doc_texts, top_k=len(doc_texts))

    if fusion_method == "rrf":
        return _reciprocal_rank_fusion(tfidf_results, bm25_results, embedding_results, top_k)
    elif fusion_method == "weighted_sum":
        return _weighted_sum_fusion(tfidf_results, bm25_results, embedding_results, weights, top_k)
    else:
        raise ValueError(f"❌ fusion method غير معرف: {fusion_method}")


def _reciprocal_rank_fusion(results1, results2, results3, top_k: int, k: int = 60) -> list:
    """
    RRF: كل وثيقة تاخذ درجة = sum(1 / (k + rank))
    """
    scores = defaultdict(float)
    
    for rank, (doc_id, _) in enumerate(results1):
        scores[doc_id] += 1 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(results2):
        scores[doc_id] += 1 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(results3):
        scores[doc_id] += 1 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def _weighted_sum_fusion(results1, results2, results3, weights: dict, top_k: int) -> list:
    """
    Weighted Sum: كل وثيقة تاخذ درجة = w1*score1 + w2*score2 + w3*score3
    بعد normalize الدرجات
    """
    def normalize(results):
        if not results:
            return {}
        max_score = max(s for _, s in results)
        min_score = min(s for _, s in results)
        diff = max_score - min_score or 1
        return {doc_id: (score - min_score) / diff for doc_id, score in results}

    norm1 = normalize(results1)
    norm2 = normalize(results2)
    norm3 = normalize(results3)

    all_docs = set(norm1) | set(norm2) | set(norm3)
    scores = {}
    for doc_id in all_docs:
        scores[doc_id] = (
            weights['tfidf'] * norm1.get(doc_id, 0) +
            weights['bm25'] * norm2.get(doc_id, 0) +
            weights['embedding'] * norm3.get(doc_id, 0)
        )

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]