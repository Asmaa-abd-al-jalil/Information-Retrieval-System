import os
import pickle
from collections import defaultdict

from services.preprocessing_service import preprocess_text
from services.database_service import init_db, insert_documents

INDEX_DIR = "data/indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

INVERTED_INDEX_PATH = os.path.join(INDEX_DIR, "inverted_index.pkl")
DOC_LENGTHS_PATH = os.path.join(INDEX_DIR, "doc_lengths.pkl")
DOC_COUNT_PATH = os.path.join(INDEX_DIR, "doc_count.pkl")

_CACHED_INVERTED_INDEX = None
_CACHED_DOC_LENGTHS = None
_CACHED_DOC_COUNT = None


def load_index():
    global _CACHED_INVERTED_INDEX, _CACHED_DOC_LENGTHS, _CACHED_DOC_COUNT

    if _CACHED_INVERTED_INDEX is not None:
        return _CACHED_INVERTED_INDEX, _CACHED_DOC_LENGTHS, _CACHED_DOC_COUNT

    if not os.path.exists(INVERTED_INDEX_PATH):
        raise FileNotFoundError("Index files missing. Build index first.")

    with open(INVERTED_INDEX_PATH, 'rb') as f:
        _CACHED_INVERTED_INDEX = pickle.load(f)

    with open(DOC_LENGTHS_PATH, 'rb') as f:
        _CACHED_DOC_LENGTHS = pickle.load(f)

    with open(DOC_COUNT_PATH, 'rb') as f:
        _CACHED_DOC_COUNT = pickle.load(f)["doc_count"]

    return _CACHED_INVERTED_INDEX, _CACHED_DOC_LENGTHS, _CACHED_DOC_COUNT


def save_index(inverted_index, doc_lengths, doc_count):
    global _CACHED_INVERTED_INDEX, _CACHED_DOC_LENGTHS, _CACHED_DOC_COUNT

    with open(INVERTED_INDEX_PATH, 'wb') as f:
        pickle.dump(inverted_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(DOC_LENGTHS_PATH, 'wb') as f:
        pickle.dump(doc_lengths, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(DOC_COUNT_PATH, 'wb') as f:
        pickle.dump({"doc_count": doc_count}, f, protocol=pickle.HIGHEST_PROTOCOL)

    _CACHED_INVERTED_INDEX = inverted_index
    _CACHED_DOC_LENGTHS = doc_lengths
    _CACHED_DOC_COUNT = doc_count



def build_inverted_index(dataset, max_docs: int = None):
    init_db()

    inverted_index = defaultdict(dict)
    doc_lengths = {}
    doc_count = 0

    raw_docs_batch = {}
    BATCH_SIZE = 5000

    for i, doc in enumerate(dataset.docs_iter()):

        if max_docs and i >= max_docs:
            break

        text = " ".join(filter(None, [
            getattr(doc, "title", ""),
            getattr(doc, "condition", ""),
            getattr(doc, "summary", ""),
            getattr(doc, "detailed_description", ""),
            getattr(doc, "eligibility", "")
        ])).strip()

        result = preprocess_text(text)
        tokens = result['final_tokens']

        if not tokens:
            continue

        tf_counts = defaultdict(int)
        for token in tokens:
            tf_counts[token] += 1

        current_doc_id = doc.doc_id

        for token, count in tf_counts.items():
            inverted_index[token][current_doc_id] = count

        doc_lengths[current_doc_id] = len(tokens)

        # store clean text (NOT doc.text)
        raw_docs_batch[current_doc_id] = text

        doc_count += 1

        if doc_count % BATCH_SIZE == 0:
            insert_documents(raw_docs_batch)
            raw_docs_batch = {}

    if raw_docs_batch:
        insert_documents(raw_docs_batch)

    return dict(inverted_index), doc_lengths, doc_count


def filter_index_terms(inverted_index: dict, doc_count: int,
                        min_df: int = 2, max_df_ratio: float = 0.9) -> dict:

    max_df = int(doc_count * max_df_ratio)
    filtered = {}

    for term, postings in inverted_index.items():
        df = len(postings)

        if df < min_df or df > max_df:
            continue

        filtered[term] = postings

    return filtered


def build_and_filter_index(dataset, max_docs: int = None,
                           min_df: int = 2, max_df_ratio: float = 0.9):

    inverted_index, doc_lengths, doc_count = build_inverted_index(dataset, max_docs)

    filtered_index = filter_index_terms(
        inverted_index,
        doc_count,
        min_df,
        max_df_ratio
    )

    save_index(filtered_index, doc_lengths, doc_count)

    return filtered_index, doc_lengths, doc_count


def get_index_stats(inverted_index, doc_lengths, doc_count):
    avg_dl = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 0

    return {
        "doc_count": doc_count,
        "unique_terms": len(inverted_index),
        "avg_doc_length": round(avg_dl, 2)
    }


def clear_index_cache():
    global _CACHED_INVERTED_INDEX, _CACHED_DOC_LENGTHS, _CACHED_DOC_COUNT
    _CACHED_INVERTED_INDEX = None
    _CACHED_DOC_LENGTHS = None
    _CACHED_DOC_COUNT = None