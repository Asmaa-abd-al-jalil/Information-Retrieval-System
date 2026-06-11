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