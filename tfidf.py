import math
from collections import Counter, defaultdict


def compute_tf(doc):
    """Compute normalized TF for one document"""
    words = doc.lower().split()
    total_terms = len(words)
    counts = Counter(words)
    return {term: count / total_terms for term, count in counts.items()}


def compute_idf(docs):
    """Compute IDF for all terms across documents"""
    N = len(docs)
    df = defaultdict(int)
    for doc in docs:
        for term in set(doc.lower().split()):
            df[term] += 1
    return {term: math.log(N / (1 + freq)) for term, freq in df.items()}


def compute_tfidf(docs):
    """Compute TF-IDF matrix for all documents"""
    idf = compute_idf(docs)
    tfidf_matrix = []

    for doc in docs:
        tf = compute_tf(doc)
        tfidf = {term: tf_val * idf[term] for term, tf_val in tf.items()}
        tfidf_matrix.append(tfidf)

    return tfidf_matrix


# Example usage
documents = ["cat eats fish", "dog eats fish", "cat chases dog"]

tfidf = compute_tfidf(documents)

print("TF-IDF Scores:")
for i, doc_scores in enumerate(tfidf, start=1):
    print(f"Document {i}:", {k: round(v, 3) for k, v in doc_scores.items()})
