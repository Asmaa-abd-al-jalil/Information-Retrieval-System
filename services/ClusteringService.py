from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class ClusteringService:
    def __init__(self, n_clusters=3):
        self.default_n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def cluster(self, documents):
        n_samples = len(documents)
        
        n_clusters = min(self.default_n_clusters, n_samples)
        
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        clusters = model.fit_predict(tfidf_matrix)
        
        output = {}
        for idx, doc in enumerate(documents):
            cluster_id = int(clusters[idx])
            output.setdefault(str(cluster_id), []).append(doc[:100])
            
        return output