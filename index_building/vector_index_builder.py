
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class VectorIndexBuilder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def build_index(self, documents):
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        return tfidf_matrix

    def query_index(self, query, index):
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = np.dot(query_vector, index.T).toarray()
        return cosine_similarities
