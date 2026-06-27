import os
import math
import numpy as np
import pickle
from gensim.models import Word2Vec
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from services.database_service import get_all_documents
from services.preprocessing_service import preprocess_text
from services.index_service import load_index
from services.ClusteringService import ClusteringService
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ==========================
# GLOBAL CACHE
# ==========================
_word2vec_model = None
_embedding_model = None

_tfidf_vectorizer = None
_tfidf_doc_matrix = None
_tfidf_doc_ids = None

_bm25_model = None
_bm25_doc_ids = None

_docs_cache = None
_doc_embedding_cache = None
_doc_embedding_ids = None
BM25_CACHE_PATH = "data/indexes/bm25.pkl"
TFIDF_CACHE_PATH = "data/indexes/tfidf.pkl"
EMBEDDINGS_CACHE_PATH = "data/indexes/doc_embeddings.pkl"
_cluster_service = ClusteringService(n_clusters=3)
# ======================================================
# DOC CACHE 
# ======================================================
def load_docs():
    global _docs_cache
    if _docs_cache is None:
        print("[CACHE] Loading documents once...")
        _docs_cache = get_all_documents()
    return _docs_cache


# ======================================================
# EMBEDDING MODEL (ONCE ONLY)
# ======================================================
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("[CACHE] Loading SentenceTransformer...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# ======================================================
# TF-IDF 
# ======================================================
def compute_tfidf_scores(query, inverted_index, doc_lengths, doc_count):

    vectorizer, doc_matrix, doc_ids = get_tfidf_model()

    query_text = preprocess_text(query)['final_text']

    query_vec = vectorizer.transform([query_text])

    scores = cosine_similarity(
        query_vec,
        doc_matrix
    ).flatten()

    return sorted(
        zip(doc_ids, scores),
        key=lambda x: x[1],
        reverse=True
    )

def get_tfidf_model():
    global _tfidf_vectorizer, _tfidf_doc_matrix, _tfidf_doc_ids

    if _tfidf_vectorizer is not None:
        return _tfidf_vectorizer, _tfidf_doc_matrix, _tfidf_doc_ids

    if os.path.exists(TFIDF_CACHE_PATH):

        print("[TFIDF] Loading cached model...")

        with open(TFIDF_CACHE_PATH, "rb") as f:
            (
                _tfidf_vectorizer,
                _tfidf_doc_matrix,
                _tfidf_doc_ids
            ) = pickle.load(f)

        return (
            _tfidf_vectorizer,
            _tfidf_doc_matrix,
            _tfidf_doc_ids
        )

    print("[TFIDF] Building model once...")

    docs = load_docs()

    _tfidf_doc_ids = list(docs.keys())

    corpus = [
        preprocess_text(text)['final_text']
        for text in docs.values()
    ]

    _tfidf_vectorizer = TfidfVectorizer()

    _tfidf_doc_matrix = _tfidf_vectorizer.fit_transform(corpus)

    os.makedirs("data/indexes", exist_ok=True)

    with open(TFIDF_CACHE_PATH, "wb") as f:
        pickle.dump(
            (
                _tfidf_vectorizer,
                _tfidf_doc_matrix,
                _tfidf_doc_ids
            ),
            f
        )

    return (
        _tfidf_vectorizer,
        _tfidf_doc_matrix,
        _tfidf_doc_ids
    )

    



# ======================================================
# BM25 
# ======================================================
def compute_bm25_scores(query, inverted_index, doc_lengths, doc_count):

    global _bm25_model, _bm25_doc_ids

    if _bm25_model is None:

        if os.path.exists(BM25_CACHE_PATH):
            print("[BM25] Loading cached model...")
            with open(BM25_CACHE_PATH, "rb") as f:
                _bm25_model, _bm25_doc_ids = pickle.load(f)

        else:
            print("[BM25] Building index once...")

            docs = load_docs()
            _bm25_doc_ids = list(docs.keys())

            corpus = [
                preprocess_text(text)['final_tokens']
                for text in docs.values()
            ]

            _bm25_model = BM25Okapi(corpus)

            os.makedirs("data/indexes", exist_ok=True)

            with open(BM25_CACHE_PATH, "wb") as f:
                pickle.dump((_bm25_model, _bm25_doc_ids), f)

    query_tokens = preprocess_text(query)['final_tokens']
    scores = _bm25_model.get_scores(query_tokens)

    return sorted(zip(_bm25_doc_ids, scores), key=lambda x: x[1], reverse=True)


# ======================================================
# EMBEDDINGS 
# ======================================================
_doc_embedding_cache = None

def load_doc_embeddings(doc_texts):

    global _doc_embedding_cache

    if _doc_embedding_cache is not None:
        return _doc_embedding_cache

    if os.path.exists(EMBEDDINGS_CACHE_PATH):

        print("[BERT] Loading cached embeddings...")

        with open(EMBEDDINGS_CACHE_PATH, "rb") as f:
            _doc_embedding_cache = pickle.load(f)

        return _doc_embedding_cache

    print("[BERT] Building embeddings once...")

    model = get_embedding_model()

    _doc_embedding_cache = {}

    for doc_id, text in doc_texts.items():
        _doc_embedding_cache[doc_id] = model.encode(
            text,
            convert_to_numpy=True
        )

    os.makedirs("data/indexes", exist_ok=True)

    with open(EMBEDDINGS_CACHE_PATH, "wb") as f:
        pickle.dump(_doc_embedding_cache, f)

    return _doc_embedding_cache

def compute_embedding_scores(query, doc_texts, top_k=10):

    model = get_embedding_model()

    query_vec = model.encode(query, convert_to_numpy=True)

    embeddings = load_doc_embeddings(doc_texts)

    scores = []

    for doc_id, text in doc_texts.items():

        if doc_id not in embeddings:
            continue

        sim = cosine_similarity(
            query_vec.reshape(1, -1),
            embeddings[doc_id].reshape(1, -1)
        )[0][0]

        scores.append((doc_id, float(sim)))

    return sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )[:top_k]


# ======================================================
# HYBRID 
# ======================================================
def hybrid_serial(original_query, doc_texts, top_k=10):

    inverted_index, doc_lengths, doc_count = load_index()

    bm25_results = compute_bm25_scores(original_query, inverted_index, doc_lengths, doc_count)

    top_50 = [d for d, _ in bm25_results[:50]]
    if doc_texts is None:
     doc_texts = load_docs()
    filtered = {d: doc_texts[d] for d in top_50 if d in doc_texts}

    return compute_embedding_scores(original_query, filtered, top_k)


def hybrid_parallel(original_query, doc_texts, top_k=10, fusion_method="rrf", weights=None):

    if weights is None:
        weights = {'tfidf': 0.25, 'bm25': 0.25, 'bert': 0.25, 'word2vec': 0.25}

    inverted_index, doc_lengths, doc_count = load_index()

    bm25_results = compute_bm25_scores(original_query, inverted_index, doc_lengths, doc_count)

    top_ids = [d for d, _ in bm25_results[:300]]
    if doc_texts is None:
     doc_texts = load_docs()
    filtered = {d: doc_texts[d] for d in top_ids if d in doc_texts}

    tfidf_results = compute_tfidf_scores(original_query, inverted_index, doc_lengths, doc_count)

    bert_results = compute_embedding_scores(original_query, filtered, len(filtered))

    w2v_results = compute_word2vec_scores(original_query, filtered, len(filtered))

    if fusion_method == "weighted_sum":
     return _weighted(tfidf_results, bm25_results, bert_results, w2v_results, weights, top_k)

    return _weighted(tfidf_results, bm25_results, bert_results, w2v_results, weights, top_k)


# ======================================================
# FUSION 
# ======================================================
def _rrf(*args, top_k, k=60):
    scores = defaultdict(float)

    for results in args[:-1]:
        for rank, (doc_id, _) in enumerate(results):
            scores[doc_id] += 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def _weighted(r1, r2, r3, r4, weights, top_k):

    def norm(r):
        if not r:
            return {}
        vals = [v for _, v in r]
        mn, mx = min(vals), max(vals)
        d = (mx - mn) or 1
        return {i: (v - mn) / d for i, v in r}

    n1, n2, n3, n4 = map(norm, [r1, r2, r3, r4])

    all_docs = set(n1) | set(n2) | set(n3) | set(n4)

    scores = {
        d: weights['tfidf'] * n1.get(d, 0) +
           weights['bm25'] * n2.get(d, 0) +
           weights['bert'] * n3.get(d, 0) +
           weights['word2vec'] * n4.get(d, 0)
        for d in all_docs
    }

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

def hybrid_clustered(original_query, doc_texts, top_k=10):

    results = hybrid_parallel(
        original_query,
        doc_texts,
        top_k=30,  
        fusion_method="weighted_sum"
    )

    top_docs = results[:30]

    diversified = _cluster_service.diversify_results(
        top_docs,
        doc_texts,
        top_k=top_k
    )

    return diversified


def compute_word2vec_scores(query: str, doc_texts: dict, top_k: int = 10):

    if not doc_texts:
        return []

    query_tokens = preprocess_text(query)['final_tokens']
    query_vec = get_word2vec_vector(query_tokens)

    scores = []

    for doc_id, text in doc_texts.items():

        tokens = preprocess_text(text)['final_tokens']
        doc_vec = get_word2vec_vector(tokens)

        sim = cosine_similarity(
            query_vec.reshape(1, -1),
            doc_vec.reshape(1, -1)
        )[0][0]

        scores.append((doc_id, float(sim)))

    return sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )[:top_k] 


def load_word2vec():
    global _word2vec_model
    if _word2vec_model is None:
        print("[WORD2VEC] Loading model...")
        _word2vec_model = Word2Vec.load("data/word2vec.model")
    return _word2vec_model


def get_word2vec_vector(tokens):
    model = load_word2vec()

    vecs = []
    for t in tokens:
        if t in model.wv:
            vecs.append(model.wv[t])

    if not vecs:
        return np.zeros(model.vector_size)

    return np.mean(vecs, axis=0)     
def compute_word2vec_scores(query: str, doc_texts: dict, top_k: int = 10) -> list:
    query_vec = get_word2vec_vector(preprocess_text(query)['final_tokens'])
    scores = [(doc_id, float(cosine_similarity_matrix(query_vec, get_word2vec_vector(preprocess_text(text)['final_tokens']).reshape(1, -1))[0])) for doc_id, text in doc_texts.items()]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

def retrieve(query: str, top_k: int = 10):
    inverted_index, doc_lengths, doc_count = load_index()

    results = compute_bm25_scores(
        query,
        inverted_index,
        doc_lengths,
        doc_count
    )

    return results[:top_k]
