import os
import math
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from services.preprocessing_service import preprocess_text
from services.index_service import load_index

os.environ["TRANSFORMERS_OFFLINE"] = "1"

_word2vec_model = None
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        model_name = 'all-MiniLM-L6-v2'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(os.path.dirname(current_dir), 'models')
        print("Downloading/Loading model...")
        _embedding_model = SentenceTransformer(model_name, cache_folder=model_path)
    return _embedding_model

def cosine_similarity_matrix(query_vec, doc_matrix):
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(len(doc_matrix))
    doc_norms = np.linalg.norm(doc_matrix, axis=1)
    doc_norms[doc_norms == 0] = 1.0
    dot_products = np.dot(doc_matrix, query_vec)
    return dot_products / (query_norm * doc_norms)

def compute_tfidf_scores(query: str, inverted_index: dict, doc_lengths: dict, doc_count: int) -> list:
    result = preprocess_text(query)
    query_tokens = result['final_tokens']
    scores = defaultdict(float)
    for token in query_tokens:
        if token not in inverted_index: continue
        df = len(inverted_index[token])
        idf = math.log((doc_count + 1) / (df + 1)) + 1
        for doc_id, tf in inverted_index[token].items():
            tf_norm = 1 + math.log(tf) if tf > 0 else 0
            scores[doc_id] += tf_norm * idf
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def compute_bm25_scores(query: str, inverted_index: dict, doc_lengths: dict, 
                       doc_count: int, k1: float = 1.5, b: float = 0.75) -> list:
    result = preprocess_text(query)
    query_tokens = result['final_tokens']
    avg_dl = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 1
    scores = defaultdict(float)
    for token in query_tokens:
        if token not in inverted_index: continue
        df = len(inverted_index[token])
        idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)
        for doc_id, tf in inverted_index[token].items():
            dl = doc_lengths.get(doc_id, avg_dl)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            scores[doc_id] += idf * tf_norm
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def compute_embedding_scores(original_query: str, doc_texts: dict, top_k: int = 10) -> list:
    if not doc_texts: return []
    model = get_embedding_model()
    query_vec = model.encode(original_query, convert_to_numpy=True)
    doc_ids = list(doc_texts.keys())
    doc_vecs = model.encode(list(doc_texts.values()), convert_to_numpy=True, show_progress_bar=False)
    sims = cosine_similarity_matrix(query_vec, doc_vecs)
    scores = [(doc_id, float(score)) for doc_id, score in zip(doc_ids, sims)]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

def hybrid_serial(original_query: str, doc_texts: dict, top_k: int = 10) -> list:
    inverted_index, doc_lengths, doc_count = load_index()
    bm25_results = compute_bm25_scores(original_query, inverted_index, doc_lengths, doc_count)
    top_50_ids = [doc_id for doc_id, _ in bm25_results[:50]]
    filtered_texts = {doc_id: doc_texts[doc_id] for doc_id in top_50_ids if doc_id in doc_texts}
    return compute_embedding_scores(original_query, filtered_texts, top_k=top_k)

def hybrid_parallel(original_query: str, doc_texts: dict, top_k: int = 10, fusion_method: str = "rrf", weights: dict = None) -> list:
    if weights is None: weights = {'tfidf': 0.25, 'bm25': 0.25, 'bert': 0.25, 'word2vec': 0.25}
    inverted_index, doc_lengths, doc_count = load_index()
    
    # تحسين الأداء: استخدام BM25 لتصفية أفضل 100 مستند للعمل عليها فقط
    bm25_results = compute_bm25_scores(original_query, inverted_index, doc_lengths, doc_count)
    top_100_ids = [doc_id for doc_id, _ in bm25_results[:100]]
    filtered_texts = {doc_id: doc_texts[doc_id] for doc_id in top_100_ids if doc_id in doc_texts}

    tfidf_results = compute_tfidf_scores(original_query, inverted_index, doc_lengths, doc_count)
    bert_results = compute_embedding_scores(original_query, filtered_texts, top_k=len(filtered_texts))
    w2v_results = compute_word2vec_scores(original_query, filtered_texts, top_k=len(filtered_texts))

    if fusion_method == "rrf":
        return _reciprocal_rank_fusion_4(tfidf_results, bm25_results, bert_results, w2v_results, top_k)
    return _weighted_sum_fusion_4(tfidf_results, bm25_results, bert_results, w2v_results, weights, top_k)

def _reciprocal_rank_fusion_4(r1, r2, r3, r4, top_k: int, k: int = 60) -> list:
    scores = defaultdict(float)
    for results in [r1, r2, r3, r4]:
        for rank, (doc_id, _) in enumerate(results):
            scores[doc_id] += 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

def _weighted_sum_fusion_4(r1, r2, r3, r4, weights, top_k):
    def normalize(results):
        if not results: return {}
        scores_list = [s for _, s in results]
        max_s, min_s = max(scores_list), min(scores_list)
        diff = max_s - min_s or 1
        return {doc_id: (score - min_s) / diff for doc_id, score in results}
    n1, n2, n3, n4 = normalize(r1), normalize(r2), normalize(r3), normalize(r4)
    all_docs = set(n1) | set(n2) | set(n3) | set(n4)
    scores = {d: weights['tfidf']*n1.get(d, 0) + weights['bm25']*n2.get(d, 0) + weights['bert']*n3.get(d, 0) + weights['word2vec']*n4.get(d, 0) for d in all_docs}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

def train_word2vec(dataset, max_docs=None):
    global _word2vec_model
    if os.path.exists("data/word2vec.model"):
        load_word2vec(); return _word2vec_model
    sentences = [preprocess_text(doc.text)['final_tokens'] for i, doc in enumerate(dataset.docs_iter()) if (not max_docs or i < max_docs) and preprocess_text(doc.text)['final_tokens']]
    _word2vec_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=2, workers=4, epochs=5)
    save_word2vec(); return _word2vec_model

def load_word2vec(path="data/word2vec.model"):
    global _word2vec_model
    if os.path.exists(path): _word2vec_model = Word2Vec.load(path)

def save_word2vec(path="data/word2vec.model"):
    os.makedirs("data", exist_ok=True)
    if _word2vec_model: _word2vec_model.save(path)

def get_word2vec_model():
    global _word2vec_model
    if _word2vec_model is None: load_word2vec()
    if _word2vec_model is None: raise ValueError("Word2Vec model not trained")
    return _word2vec_model

def get_word2vec_vector(tokens):
    model = get_word2vec_model()
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def compute_word2vec_scores(query: str, doc_texts: dict, top_k: int = 10) -> list:
    query_vec = get_word2vec_vector(preprocess_text(query)['final_tokens'])
    scores = [(doc_id, float(cosine_similarity_matrix(query_vec, get_word2vec_vector(preprocess_text(text)['final_tokens']).reshape(1, -1))[0])) for doc_id, text in doc_texts.items()]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]