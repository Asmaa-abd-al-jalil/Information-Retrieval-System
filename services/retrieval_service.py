"""
retrieval_service.py — نسخة موفّرة للـ RAM
============================================
المشكلة السابقة: كنا نحمّل 190K وثيقة كاملة بالـ RAM لبناء BM25 و TF-IDF
الحل: نشتغل مباشرة من الـ inverted index المبني مسبقاً
  - BM25  : من الـ inverted index (TF + DF جاهزين)
  - TF-IDF: من الـ inverted index (نفس الفكرة)
  - Embedding: على top-K من BM25 فقط (مو كل الوثائق)
"""

import os
import math
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from services.preprocessing_service import preprocess_text
from services.index_service import load_index
from services.database_service import get_documents_by_ids

os.environ["TRANSFORMERS_OFFLINE"] = "1"

_embedding_model = None
_word2vec_model  = None


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def _cosine(q: np.ndarray, D: np.ndarray) -> np.ndarray:
    qn = np.linalg.norm(q)
    if qn == 0:
        return np.zeros(len(D))
    dn = np.linalg.norm(D, axis=1)
    dn[dn == 0] = 1.0
    return D.dot(q) / (qn * dn)


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        name  = "all-MiniLM-L6-v2"
        cache = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "models")
        print("[BERT] Loading SentenceTransformer...")
        _embedding_model = SentenceTransformer(name, cache_folder=cache)
    return _embedding_model


# ─────────────────────────────────────────
# BM25 — من الـ inverted index مباشرة
# RAM: فقط postings للـ query terms
# ─────────────────────────────────────────

def compute_bm25_scores(
    query: str,
    inverted_index: dict = None,
    doc_lengths: dict = None,
    doc_count: int = None,
    k1: float = 1.5,
    b:  float = 0.75,
) -> list:
    if inverted_index is None:
        inverted_index, doc_lengths, doc_count = load_index()

    avg_dl   = sum(doc_lengths.values()) / max(doc_count, 1)
    q_tokens = preprocess_text(query)["final_tokens"]
    if not q_tokens:
        return []

    scores = defaultdict(float)
    for token in q_tokens:
        if token not in inverted_index:
            continue
        postings = inverted_index[token]
        df       = len(postings)
        idf      = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)
        for doc_id, tf in postings.items():
            dl    = doc_lengths.get(doc_id, avg_dl)
            denom = tf + k1 * (1 - b + b * dl / avg_dl)
            scores[doc_id] += idf * (tf * (k1 + 1)) / denom

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────
# TF-IDF — من الـ inverted index مباشرة
# ─────────────────────────────────────────

def compute_tfidf_scores(
    query: str,
    inverted_index: dict = None,
    doc_lengths: dict = None,
    doc_count: int = None,
) -> list:
    if inverted_index is None:
        inverted_index, doc_lengths, doc_count = load_index()

    q_tokens = preprocess_text(query)["final_tokens"]
    if not q_tokens:
        return []

    q_tf = defaultdict(int)
    for t in q_tokens:
        q_tf[t] += 1

    scores    = defaultdict(float)
    q_norm_sq = 0.0

    for token, q_count in q_tf.items():
        if token not in inverted_index:
            continue
        postings = inverted_index[token]
        df       = len(postings)
        idf      = math.log((doc_count + 1) / (df + 1)) + 1

        q_w        = (1 + math.log(q_count)) * idf
        q_norm_sq += q_w ** 2

        for doc_id, tf in postings.items():
            dl   = doc_lengths.get(doc_id, 1)
            d_w  = (1 + math.log(max(tf, 1))) * idf / max(math.sqrt(dl), 1)
            scores[doc_id] += q_w * d_w

    q_norm = math.sqrt(q_norm_sq) or 1.0
    final  = {did: s / q_norm for did, s in scores.items()}
    return sorted(final.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────
# BERT EMBEDDING — على candidates فقط
# ─────────────────────────────────────────

def compute_embedding_scores(
    query: str,
    doc_texts: dict,
    top_k: int = 10,
) -> list:
    if not doc_texts:
        return []
    model    = get_embedding_model()
    doc_ids  = list(doc_texts.keys())
    doc_vecs = model.encode(list(doc_texts.values()),
                            convert_to_numpy=True,
                            show_progress_bar=False,
                            batch_size=64)
    q_vec = model.encode(query, convert_to_numpy=True)
    sims  = _cosine(q_vec, doc_vecs)
    return sorted(zip(doc_ids, sims.tolist()),
                  key=lambda x: x[1], reverse=True)[:top_k]


# ─────────────────────────────────────────
# WORD2VEC
# ─────────────────────────────────────────

def load_word2vec(path: str = "data/word2vec.model"):
    global _word2vec_model
    if os.path.exists(path):
        _word2vec_model = Word2Vec.load(path)
        print("[Word2Vec] Loaded.")

def save_word2vec(path: str = "data/word2vec.model"):
    os.makedirs("data", exist_ok=True)
    if _word2vec_model:
        _word2vec_model.save(path)

def get_word2vec_model() -> Word2Vec:
    global _word2vec_model
    if _word2vec_model is None:
        load_word2vec()
    if _word2vec_model is None:
        raise ValueError("Word2Vec not trained.")
    return _word2vec_model

def _w2v_vec(tokens):
    m    = get_word2vec_model()
    vecs = [m.wv[t] for t in tokens if t in m.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(m.vector_size)

def compute_word2vec_scores(query: str, doc_texts: dict, top_k: int = 10) -> list:
    q_vec   = _w2v_vec(preprocess_text(query)["final_tokens"])
    results = []
    for doc_id, text in doc_texts.items():
        d_vec  = _w2v_vec(preprocess_text(text)["final_tokens"])
        qn, dn = np.linalg.norm(q_vec), np.linalg.norm(d_vec)
        sim    = float(np.dot(q_vec, d_vec) / (qn * dn)) if qn and dn else 0.0
        results.append((doc_id, sim))
    return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

def train_word2vec(dataset, max_docs=None):
    global _word2vec_model
    if os.path.exists("data/word2vec.model"):
        load_word2vec()
        return _word2vec_model

    print("[Word2Vec] Training on full dataset (streaming)...")
    sentences = []
    
    # تحديد عينة ذكية وسريعة بـ 10,000 وثيقة لتنتهي فوراً
    limit = 10000
    print(f"[Word2Vec] جاري تدريب الموديل على عينة سريعة من {limit} وثيقة...")

    for i, doc in enumerate(dataset.docs_iter()):
        if i >= limit:
            break
            
        text = " ".join(filter(None, [
            getattr(doc, "title", ""),
            getattr(doc, "condition", ""),
            getattr(doc, "summary", ""),
            getattr(doc, "detailed_description", ""),
            getattr(doc, "eligibility", ""),
        ]))
        tokens = preprocess_text(text)["final_tokens"]
        if tokens:
            sentences.append(tokens)
            
        # جعل العداد يطبع ويتحرك بسرعة كل 1000 وثيقة بدلاً من 20 ألف!
        if i % 1000 == 0:
            print(f"[⚙️ تقدم سريع] تم تنظيف ومعالجة {i} / {limit} مستند طبي...")

    _word2vec_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=2, workers=4, epochs=5)
    save_word2vec()
    print(f"[Word2Vec] Done. {len(sentences)} docs.")
    return _word2vec_model


# ─────────────────────────────────────────
# HYBRID SERIAL
# ─────────────────────────────────────────

def hybrid_serial(query: str, doc_texts: dict = None,
                  top_k: int = 10, bm25_candidates: int = 200) -> list:
    inverted_index, doc_lengths, doc_count = load_index()
    bm25_results = compute_bm25_scores(query, inverted_index, doc_lengths, doc_count)

    effective = max(bm25_candidates, top_k * 3)
    top_ids   = [did for did, _ in bm25_results[:effective]]

    if doc_texts:
        filtered = {did: doc_texts[did] for did in top_ids if did in doc_texts}
    else:
        filtered = get_documents_by_ids(top_ids)

    return compute_embedding_scores(query, filtered, top_k=top_k)


# ─────────────────────────────────────────
# HYBRID PARALLEL
# ─────────────────────────────────────────

def hybrid_parallel(query, top_k=1000, fusion_method="rrf", weights=None):
    from services.database_service import get_documents_by_ids

    tfidf_results = compute_tfidf_scores(query, top_k=1000)
    bm25_results  = compute_bm25_scores(query, top_k=1000)

    # ترشيح أفضل 300 وثيقة من BM25 لإخضاعها للفهم العميق والذكاء الاصطناعي
    candidate_ids = [did for did, _ in bm25_results[:300]]
    
    # جلب النصوص الكاملة للـ 300 وثيقة فوراً وبقوة من قاعدة البيانات التي ملأناها
    doc_texts = get_documents_by_ids(candidate_ids)
    print(f"[INFO] Loaded {len(doc_texts)} documents from DB")

    # حساب تشابه المعنى الطبي عبر الـ BERT والـ Word2Vec
    bert_results = compute_embedding_scores(query, doc_texts, top_k=len(doc_texts))
    w2v_results  = compute_word2vec_scores(query,  doc_texts, top_k=len(doc_texts))

    all_r = [tfidf_results, bm25_results, bert_results, w2v_results]
    return _rrf(all_r, top_k) if fusion_method == "rrf" else _weighted_sum(all_r, weights, top_k)


def _rrf(result_lists, top_k, k=60):
    scores = defaultdict(float)
    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def _weighted_sum(result_lists, weights, top_k):
    keys = ["tfidf", "bm25", "bert", "word2vec"]

    def norm(r):
        if not r: return {}
        vals = [s for _, s in r]
        lo, hi = min(vals), max(vals)
        d = (hi - lo) or 1.0
        return {did: (s - lo) / d for did, s in r}

    normed  = [norm(r) for r in result_lists]
    all_ids = set().union(*[n.keys() for n in normed])
    final   = {
        did: sum(weights.get(keys[i], 0) * normed[i].get(did, 0)
                 for i in range(len(result_lists)))
        for did in all_ids
    }
    return sorted(final.items(), key=lambda x: x[1], reverse=True)[:top_k]