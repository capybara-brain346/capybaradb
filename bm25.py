import math
from collections import Counter


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [doc.lower().split() for doc in docs]
        self.N = len(docs)
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(doc) for doc in self.docs) / self.N

        # Precompute document frequencies
        self.df = {}
        for doc in self.docs:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    def idf(self, term):
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query, doc_index):
        doc = self.docs[doc_index]
        doc_len = len(doc)
        tf = Counter(doc)

        score = 0.0
        for term in query.lower().split():
            if term not in tf:
                continue
            freq = tf[term]
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (
                1 - self.b + self.b * (doc_len / self.avgdl)
            )
            score += self.idf(term) * (numerator / denominator)
        return score

    def rank(self, query):
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        return sorted(scores, key=lambda x: x[1], reverse=True)


# Example usage
documents = ["cat eats fish", "dog eats fish", "cat chases dog"]

bm25 = BM25(documents)

print("Ranking for query: 'cat fish'")
for doc_id, score in bm25.rank("cat fish"):
    print(f"Doc {doc_id + 1}: Score = {round(score, 3)}")
