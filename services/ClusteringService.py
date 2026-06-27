from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class ClusteringService:

    def __init__(self, n_clusters=3):
        self.default_n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def cluster(self, documents):

        n_samples = len(documents)
        n_clusters = min(self.default_n_clusters, n_samples)

        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        tfidf_matrix = self.vectorizer.fit_transform(documents)
        clusters = model.fit_predict(tfidf_matrix)

        output = {}

        for idx, doc in enumerate(documents):
            cluster_id = int(clusters[idx])
            output.setdefault(str(cluster_id), []).append(doc[:100])

        return output

    def diversify_results(self, ranked_results, doc_texts, top_k=10):

        if not ranked_results:
            return []

        candidate_docs = ranked_results[:50]

        ids = [doc_id for doc_id, _ in candidate_docs]

        texts = [
            doc_texts.get(doc_id, "")
            for doc_id in ids
        ]

        if len(texts) <= self.default_n_clusters:
            return candidate_docs[:top_k]

        tfidf_matrix = self.vectorizer.fit_transform(texts)

        model = KMeans(
            n_clusters=min(self.default_n_clusters, len(texts)),
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(tfidf_matrix)

        diversified = []
        used_clusters = set()

        for idx, cluster_id in enumerate(labels):

            if cluster_id not in used_clusters:
                diversified.append(candidate_docs[idx])
                used_clusters.add(cluster_id)

            if len(diversified) >= top_k:
                break

        for doc in candidate_docs:

            if doc not in diversified:
                diversified.append(doc)

            if len(diversified) >= top_k:
                break

        return diversified[:top_k]