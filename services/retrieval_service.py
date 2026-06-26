import os
import math
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from services.database_service import get_all_documents
from services.preprocessing_service import preprocess_text
from services.index_service import load_index

os.environ["TRANSFORMERS_OFFLINE"] = "1"

_word2vec_model = None
_embedding_model = None
_tfidf_vectorizer = None
_tfidf_doc_matrix = None

_bm25_model = None
_bm25_doc_ids = None
_bm25_corpus_ready = False

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

def compute_tfidf_scores(query: str,
                          inverted_index: dict,
                          doc_lengths: dict,
                          doc_count: int) -> list:

    global _tfidf_vectorizer, _tfidf_doc_matrix

    docs = get_all_documents()

    if not docs:
        return []

    doc_ids = list(docs.keys())

    corpus = [
        preprocess_text(text)['final_text']
        for text in docs.values()
    ]

    if _tfidf_vectorizer is None:
        print("Building TF-IDF model...")

        _tfidf_vectorizer = TfidfVectorizer()

        _tfidf_doc_matrix = (
            _tfidf_vectorizer.fit_transform(corpus)
        )

    processed_query = preprocess_text(
        query
    )['final_text']

    query_vector = _tfidf_vectorizer.transform(
        [processed_query]
    )

    similarities = cosine_similarity(
        query_vector,
        _tfidf_doc_matrix
    ).flatten()

    scores = list(zip(doc_ids, similarities))

    return sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

def compute_bm25_scores(
    query: str,
    inverted_index: dict,
    doc_lengths: dict,
    doc_count: int
) -> list:

    global _bm25_model, _bm25_doc_ids, _bm25_corpus_ready

    if _bm25_model is None:

        print("[BM25] Loading documents from DB...")

        docs = get_all_documents()

        if not docs:
            return []

        _bm25_doc_ids = list(docs.keys())

        print("[BM25] Preprocessing corpus (ONE TIME)...")

        corpus = []

        for i, text in enumerate(docs.values()):

            # تقدم progress لتتأكدي أنه لا يتوقف
            if i % 20000 == 0:
                print(f"[BM25] processed {i} docs")

            tokens = preprocess_text(text)['final_tokens']
            corpus.append(tokens)

        print("[BM25] Building BM25 index...")

        _bm25_model = BM25Okapi(corpus)

        _bm25_corpus_ready = True

    # --------- query processing ----------
    query_tokens = preprocess_text(query)['final_tokens']

    scores = _bm25_model.get_scores(query_tokens)

    results = list(zip(_bm25_doc_ids, scores))

    return sorted(results, key=lambda x: x[1], reverse=True)

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
    
    bm25_results = compute_bm25_scores(original_query, inverted_index, doc_lengths, doc_count)
    top_300_ids = [
    doc_id for doc_id, _
    in bm25_results[:300]]

    filtered_texts = {
    doc_id: doc_texts[doc_id]
    for doc_id in top_300_ids
    if doc_id in doc_texts}

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
        load_word2vec()
        return _word2vec_model

    print("Training Word2Vec model...")

    sentences = []

    for i, doc in enumerate(dataset.docs_iter()):

        if max_docs and i >= max_docs:
            break

        full_text = (
            f"{doc.title} "
            f"{doc.condition} "
            f"{doc.summary} "
            f"{doc.detailed_description} "
            f"{doc.eligibility}"
        )

        tokens = preprocess_text(full_text)['final_tokens']

        if tokens:
            sentences.append(tokens)

    print(f"Training on {len(sentences)} documents...")

    _word2vec_model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        epochs=10
    )

    save_word2vec()

    print("Word2Vec model trained and saved successfully.")

    return _word2vec_model

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