import math
from collections import Counter, defaultdict

from .base_index import BaseIndex
from .index_types import IndexType


class TFIDFIndex(BaseIndex):
    def __init__(self, docs: list[str]) -> None:
        super().__init__(docs)
        self.index_type = IndexType.TFIDF
        self.total_documents = len(self.docs)
        self.tfidf_vectors = None

    def compute_index(self):
        idf = self._compute_idf()
        vocabulary = sorted(idf.keys())
        tfidf = []

        for doc in self.docs:
            tf = self._compute_tf(doc)
            doc_vector = [tf[term] * idf[term] for term in vocabulary]
            tfidf.append(doc_vector)

        self.tfidf_vectors = tfidf
        self.vocabulary = vocabulary
        return tfidf

    def search(self, query, top_k=5):
        if self.tfidf_vectors is None:
            self.compute_index()

        query_tf = self._compute_tf(query)
        query_vector = [query_tf[term] for term in self.vocabulary]

        scores = []
        for i, doc_vector in enumerate(self.tfidf_vectors):
            score = sum(q * d for q, d in zip(query_vector, doc_vector))
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _compute_tf(self, doc: str):
        words = doc.lower().split()
        total_terms = len(words)
        frequency_map = Counter(words)
        term_frequency = defaultdict(float)

        for term in words:
            term_frequency[term] = frequency_map[term] / total_terms

        return term_frequency

    def _compute_idf(self):
        idf = defaultdict(float)
        document_frequency = defaultdict(int)

        for doc in self.docs:
            words = set(doc.lower().split())
            for term in words:
                document_frequency[term] += 1

        for term, freq in document_frequency.items():
            idf[term] = math.log(self.total_documents / freq)

        return idf


if __name__ == "__main__":
    documents = [
        """Machine learning is a subset of artificial intelligence that focuses on 
    building systems that learn from data. Machine learning algorithms use 
    statistical techniques to enable computers to improve their performance 
    on tasks through experience. Deep learning is a specialized form of 
    machine learning that uses neural networks with multiple layers.""",
        """Cooking pasta is simple and delicious. First, boil water in a large pot 
    and add salt. Then add the pasta and cook for eight to ten minutes. 
    Drain the pasta and add your favorite sauce. Italian cooking emphasizes 
    fresh ingredients and simple preparation techniques. Pasta dishes are 
    beloved around the world.""",
        """Space exploration has advanced significantly in recent decades. NASA and 
    private companies are developing new technologies for Mars missions. 
    Rockets are becoming reusable, reducing the cost of space travel. 
    Scientists study planets, stars, and galaxies to understand our universe. 
    The International Space Station orbits Earth as a research laboratory.""",
        """Climate change poses significant challenges to our planet. Rising 
    temperatures affect weather patterns and sea levels. Greenhouse gases 
    from human activities trap heat in the atmosphere. Scientists emphasize 
    the need for renewable energy and sustainable practices. Global 
    cooperation is essential to address climate change effectively.""",
        """Ancient civilizations built remarkable structures that still stand today. 
    The pyramids of Egypt showcase advanced engineering knowledge. Roman 
    aqueducts supplied water to cities across their empire. Ancient Greeks 
    made significant contributions to philosophy, mathematics, and democracy. 
    Archaeologists continue discovering artifacts that reveal secrets of 
    ancient cultures.""",
    ]

    tfidf = TFIDFIndex(documents)
    vectors = tfidf.compute_index()

    for i, doc_scores in enumerate(vectors, start=1):
        print(f"{i}:{doc_scores}")
