import os
import math
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from services.preprocessing_service import preprocess_text
from services.index_service import load_index

_word2vec_model = None
_embedding_model = None

def compute_tfidf_scores(query: str, inverted_index: dict, doc_lengths: dict, doc_count: int) -> list:
    result = preprocess_text(query)
    query_tokens = result['final_tokens']
    scores = defaultdict(float)

    for token in query_tokens:
        if token not in inverted_index:
            continue
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
        if token not in inverted_index:
            continue
        df = len(inverted_index[token])
        idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)

        for doc_id, tf in inverted_index[token].items():
            dl = doc_lengths.get(doc_id, avg_dl)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            scores[doc_id] += idf * tf_norm

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


def cosine_similarity_matrix(query_vec, doc_matrix):
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(len(doc_matrix))
    doc_norms = np.linalg.norm(doc_matrix, axis=1)
    doc_norms[doc_norms == 0] = 1.0
    dot_products = np.dot(doc_matrix, query_vec)
    return dot_products / (query_norm * doc_norms)


def compute_embedding_scores(original_query: str, doc_texts: dict, top_k: int = 10) -> list:
    model = get_embedding_model()
    query_vec = model.encode(original_query, convert_to_numpy=True)
    
    doc_ids = list(doc_texts.keys())
    doc_vecs = model.encode(list(doc_texts.values()), convert_to_numpy=True, show_progress_bar=False)
    sims = cosine_similarity_matrix(query_vec, doc_vecs)
    
    scores = [(doc_id, float(score)) for doc_id, score in zip(doc_ids, sims)]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


def hybrid_serial(original_query: str, doc_texts: dict, top_k: int = 10,
                  k1: float = 1.5, b: float = 0.75) -> list:
    inverted_index, doc_lengths, doc_count = load_index()
    
    processed_result = preprocess_text(original_query)
    processed_query = processed_result['final_text']
    
    bm25_results = compute_bm25_scores(processed_query, inverted_index, doc_lengths, doc_count, k1=k1, b=b)
    
    top_50_candidates = bm25_results[:50]
    top_50_ids = set(doc_id for doc_id, _ in top_50_candidates)
    
    filtered_texts = {doc_id: doc_texts[doc_id] for doc_id in top_50_ids if doc_id in doc_texts}
    
    if not filtered_texts:
        return top_50_candidates[:top_k]
    
    return compute_embedding_scores(original_query, filtered_texts, top_k=top_k)


def hybrid_parallel(original_query: str, doc_texts: dict, top_k: int = 10,
                    k1: float = 1.5, b: float = 0.75,
                    fusion_method: str = "rrf", weights: dict = None) -> list:
    if weights is None:
        weights = {'tfidf': 0.25, 'bm25': 0.25, 'bert': 0.25, 'word2vec': 0.25}

    inverted_index, doc_lengths, doc_count = load_index()
    processed_result = preprocess_text(original_query)
    processed_query = processed_result['final_text']

    tfidf_results = compute_tfidf_scores(processed_query, inverted_index, doc_lengths, doc_count)
    bm25_results = compute_bm25_scores(processed_query, inverted_index, doc_lengths, doc_count, k1=k1, b=b)
    bert_results = compute_embedding_scores(original_query, doc_texts, top_k=len(doc_texts))
    w2v_results = compute_word2vec_scores(original_query, doc_texts, top_k=len(doc_texts))

    if fusion_method == "rrf":
        return _reciprocal_rank_fusion_4(tfidf_results, bm25_results, bert_results, w2v_results, top_k)
    elif fusion_method == "weighted_sum":
        return _weighted_sum_fusion_4(tfidf_results, bm25_results, bert_results, w2v_results, weights, top_k)
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")


def _reciprocal_rank_fusion_4(r1, r2, r3, r4, top_k: int, k: int = 60) -> list:
    scores = defaultdict(float)
    for results in [r1, r2, r3, r4]:
        for rank, (doc_id, _) in enumerate(results):
            scores[doc_id] += 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def _weighted_sum_fusion_4(r1, r2, r3, r4, weights: dict, top_k: int) -> list:
    def normalize(results):
        if not results:
            return {}
        scores_list = [s for _, s in results]
        max_s = max(scores_list)
        min_s = min(scores_list)
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
    
    _word2vec_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=2, workers=4, epochs=5)
    save_word2vec()
    return _word2vec_model


def load_word2vec(path: str = "data/word2vec.model"):
    global _word2vec_model
    if os.path.exists(path):
        _word2vec_model = Word2Vec.load(path)


def save_word2vec(path: str = "data/word2vec.model"):
    os.makedirs("data", exist_ok=True)
    if _word2vec_model:
        _word2vec_model.save(path)


def get_word2vec_model():
    global _word2vec_model
    if _word2vec_model is None:
        load_word2vec()
    if _word2vec_model is None:
        raise ValueError("Word2Vec model is not trained")
    return _word2vec_model


def get_word2vec_vector(tokens: list) -> np.ndarray:
    model = get_word2vec_model()
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def compute_word2vec_scores(query: str, doc_texts: dict, top_k: int = 10) -> list:
    result = preprocess_text(query)
    query_vec = get_word2vec_vector(result['final_tokens'])
    
    scores = []
    for doc_id, text in doc_texts.items():
        doc_result = preprocess_text(text)
        doc_vec = get_word2vec_vector(doc_result['final_tokens'])
        score = cosine_similarity_matrix(query_vec, doc_vec.reshape(1, -1))[0]
        scores.append((doc_id, float(score)))
    
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]