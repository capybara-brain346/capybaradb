import math
from collections import Counter, defaultdict


class BM25Index:
    def __init__(self, docs):
        self.docs = [doc.lower().split() for doc in docs]
        self.docs_length = len(docs)
        self.avg_document_length = sum(len(doc) for doc in self.docs) / self.docs_length
        self.k = 1.2
        self.b = 0.75
        self.df = defaultdict(int)

        for doc in self.docs:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        term_df = self.df.get(term, 0)
        return math.log((self.docs_length - term_df + 0.5) / term_df + 0.5 + 1)

    def score(self, query, doc_index):
        doc = self.docs[doc_index]
        doc_len = len(doc)
        tf = Counter(doc)

        score = 0.0
        for term in query.lower().split():
            if term not in tf:
                continue
            score += (
                self.idf(term)
                * (tf[term] * (self.k + 1))
                / (
                    tf[term]
                    + self.k
                    * (1 - self.b + self.b * (doc_len / self.avg_document_length))
                )
            )
        return score

    def rank(self, query):
        scores = [(i, self.score(query, i)) for i in range(self.docs_length)]
        return sorted(scores, key=lambda x: x[1], reverse=True)


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
    bm25 = BM25Index(documents)
    score = bm25.rank("Ancient civilizations")
    print(score)
