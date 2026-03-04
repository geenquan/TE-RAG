
from collections import defaultdict
import numpy as np

class KeywordIndexBuilder:
    def __init__(self):
        self.index = defaultdict(list)

    def build_index(self, documents):
        for doc_id, document in enumerate(documents):
            terms = document.split(" ")
            for term in terms:
                self.index[term].append(doc_id)
        return self.index

    def query_index(self, query):
        query_terms = query.split(" ")
        matched_docs = []
        for term in query_terms:
            if term in self.index:
                matched_docs.extend(self.index[term])
        return list(set(matched_docs))  # 去重
