import os
import math
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
from services.preprocessing_service import preprocess_text
from services.index_service import load_index
from sentence_transformers import SentenceTransformer
from services.database_service import get_documents_by_ids

# متغير عالمي للنموذج
_word2vec_model = None

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
    print(f"  📊 BM25 Parameters: k1={k1}, b={b}")
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

def retrieve_with_text(query: str, model: str = "tfidf", top_k: int = 10,
                       k1: float = 1.5, b: float = 0.75,
                       doc_texts: dict = None,
                       fusion_method: str = "rrf") -> list:
    """
    نفس الاسترجاع بس بيرجع النص الأصلي من الـ Database
    """
    # جيب الـ IDs
    from services.query_service import search
    response = search(query, model=model, top_k=top_k,
                     k1=k1, b=b, doc_texts=doc_texts,
                     fusion_method=fusion_method)
    
    results = response['results']
    doc_ids = [doc_id for doc_id, _ in results]
    
    # اقرأ النص الأصلي من الـ Database
    raw_texts = get_documents_by_ids(doc_ids)
    
    # ادمج النتائج مع النص الأصلي
    final_results = []
    for doc_id, score in results:
        final_results.append({
            "doc_id":   doc_id,
            "score":    round(score, 4),
            "raw_text": raw_texts.get(doc_id, "")[:200]  # أول 200 حرف
        })
    
    return final_results


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
                  k1: float = 1.5, b: float = 0.75,
                  original_query: str = None) -> list:
    
    inverted_index, doc_lengths, doc_count = load_index()
    
    # المرحلة 1: TF-IDF بالنص المعالج
    print("  🔄 المرحلة 1: TF-IDF...")
    tfidf_results = compute_tfidf_scores(
        query, inverted_index, doc_lengths, doc_count
    )
    top_100_ids = set(doc_id for doc_id, _ in tfidf_results[:100])

    # المرحلة 2: BM25 بالنص المعالج
    print("  🔄 المرحلة 2: BM25...")
    bm25_results = compute_bm25_scores(
        query, inverted_index, doc_lengths, doc_count, k1=k1, b=b
    )
    top_50 = [(doc_id, score) for doc_id, score in bm25_results 
              if doc_id in top_100_ids][:50]
    top_50_ids = set(doc_id for doc_id, _ in top_50)

    # المرحلة 3: Embedding بالنص الأصلي
    print("  🔄 المرحلة 3: Embedding...")
    embed_query = original_query if original_query else query
    filtered_texts = {doc_id: doc_texts[doc_id] 
                     for doc_id in top_50_ids if doc_id in doc_texts}
    
    if not filtered_texts:
        return top_50[:top_k]
    
    return compute_embedding_scores(embed_query, filtered_texts, top_k=top_k)


def hybrid_parallel(query: str, doc_texts: dict, top_k: int = 10,
                    k1: float = 1.5, b: float = 0.75,
                    fusion_method: str = "rrf",
                    weights: dict = None) -> list:
    """
    Hybrid Parallel: TF-IDF + BM25 + BERT + Word2Vec بالتوازي
    fusion_method: 'rrf' أو 'weighted_sum'
    """
    if weights is None:
        weights = {'tfidf': 0.25, 'bm25': 0.25, 'bert': 0.25, 'word2vec': 0.25}

    inverted_index, doc_lengths, doc_count = load_index()

    print("  🔄 TF-IDF...")
    tfidf_results = compute_tfidf_scores(query, inverted_index, doc_lengths, doc_count)

    print("  🔄 BM25...")
    bm25_results = compute_bm25_scores(query, inverted_index, doc_lengths, doc_count, k1=k1, b=b)

    print("  🔄 BERT Embedding...")
    bert_results = compute_embedding_scores(query, doc_texts, top_k=len(doc_texts))

    print("  🔄 Word2Vec Embedding...")
    w2v_results = compute_word2vec_scores(query, doc_texts, top_k=len(doc_texts))

    if fusion_method == "rrf":
        return _reciprocal_rank_fusion_4(tfidf_results, bm25_results, bert_results, w2v_results, top_k)
    elif fusion_method == "weighted_sum":
        return _weighted_sum_fusion_4(tfidf_results, bm25_results, bert_results, w2v_results, weights, top_k)
    else:
        raise ValueError(f"❌ fusion method غير معرف: {fusion_method}")


def _reciprocal_rank_fusion_4(r1, r2, r3, r4, top_k: int, k: int = 60) -> list:
    """RRF لـ 4 نماذج"""
    scores = defaultdict(float)
    for results in [r1, r2, r3, r4]:
        for rank, (doc_id, _) in enumerate(results):
            scores[doc_id] += 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def _weighted_sum_fusion_4(r1, r2, r3, r4, weights: dict, top_k: int) -> list:
    """Weighted Sum لـ 4 نماذج"""
    def normalize(results):
        if not results:
            return {}
        max_s = max(s for _, s in results)
        min_s = min(s for _, s in results)
        diff = max_s - min_s or 1
        return {doc_id: (score - min_s) / diff for doc_id, score in results}

    n1 = normalize(r1)
    n2 = normalize(r2)
    n3 = normalize(r3)
    n4 = normalize(r4)

    all_docs = set(n1) | set(n2) | set(n3) | set(n4)
    scores = {}
    for doc_id in all_docs:
        scores[doc_id] = (
            weights['tfidf']    * n1.get(doc_id, 0) +
            weights['bm25']     * n2.get(doc_id, 0) +
            weights['bert']     * n3.get(doc_id, 0) +
            weights['word2vec'] * n4.get(doc_id, 0)
        )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

def train_word2vec(dataset, max_docs: int = None):
    global _word2vec_model
    print("🔄 جاري تدريب Word2Vec...")
    
    # إذا موجود على الـ disk حمّله مباشرة
    if os.path.exists("data/word2vec.model"):
        load_word2vec()
        return _word2vec_model
    
    sentences = []
    for i, doc in enumerate(dataset.docs_iter()):
        if max_docs and i >= max_docs:
            break
        result = preprocess_text(doc.text)
        if result['final_tokens']:
            sentences.append(result['final_tokens'])
    
    _word2vec_model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        epochs=5
    )
    
    # حفظ تلقائي بعد التدريب
    save_word2vec()
    print(f"✅ تم تدريب Word2Vec على {len(sentences):,} وثيقة")
    return _word2vec_model

def get_word2vec_model():
    global _word2vec_model
    if _word2vec_model is None:
        raise ValueError("❌ Word2Vec لسا ما تدرّب، شغّل train_word2vec أولاً")
    return _word2vec_model

def get_word2vec_vector(tokens: list) -> np.ndarray:
    """حساب متوسط متجهات الكلمات"""
    model = get_word2vec_model()
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def compute_word2vec_scores(query: str, doc_texts: dict, top_k: int = 10) -> list:
    """Word2Vec: حساب التشابه بين الاستعلام والوثائق"""
    result = preprocess_text(query)
    query_tokens = result['final_tokens']
    query_vec = get_word2vec_vector(query_tokens)
    
    scores = []
    for doc_id, text in doc_texts.items():
        doc_result = preprocess_text(text)
        doc_vec = get_word2vec_vector(doc_result['final_tokens'])
        score = cosine_similarity(query_vec, doc_vec)
        scores.append((doc_id, score))
    
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

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
# حفظ ال Word2Vec على الdisk
def save_word2vec(path: str = "data/word2vec.model"):
    """حفظ نموذج Word2Vec على الـ disk"""
    model = get_word2vec_model()
    os.makedirs("data", exist_ok=True)
    model.save(path)
    print(f"✅ تم حفظ Word2Vec: {path}")

def load_word2vec(path: str = "data/word2vec.model"):
    """تحميل نموذج Word2Vec من الـ disk"""
    global _word2vec_model
    if os.path.exists(path):
        _word2vec_model = Word2Vec.load(path)
        print(f"✅ تم تحميل Word2Vec من الـ disk")
    else:
        raise FileNotFoundError("❌ ما في نموذج محفوظ، شغّل train_word2vec أولاً")