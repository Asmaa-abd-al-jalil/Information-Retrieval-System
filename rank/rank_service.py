import math

def calculate_bm25(tf, df, doc_length, avg_doc_length, doc_count, k1=1.5, b=0.75):
    """
    Calculate the BM25 relevance score for a single document.
    """
    idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)
    score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
    return score

def rank_retrieved_documents(query_tokens, retrieved_docs, index, doc_lengths, doc_count):
    """
    Rank the subset of retrieved documents based on explicit query tokens.
    """
    avg_doc_length = sum(doc_lengths.values()) / doc_count if doc_count > 0 else 1
    results = []
    
    for doc_id, doc_info in retrieved_docs.items():
        score = 0
        for token in query_tokens:
            if token in index and doc_id in index[token]:
                tf = index[token][doc_id]
                df = len(index[token])
                score += calculate_bm25(tf, df, doc_lengths[doc_id], avg_doc_length, doc_count)
        results.append((doc_id, score))
    
    return sorted(results, key=lambda x: x[1], reverse=True)