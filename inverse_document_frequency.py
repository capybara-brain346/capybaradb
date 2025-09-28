import math
from collections import defaultdict


def compute_idf(docs):
    """
    Compute IDF for each term across all documents.
    docs: list of strings (documents)
    Returns dictionary: term -> idf score
    """
    N = len(docs)
    df = defaultdict(int)

    # Count in how many documents each term appears
    for doc in docs:
        unique_terms = set(doc.lower().split())
        for term in unique_terms:
            df[term] += 1

    # Compute IDF
    idf = {term: math.log(N / (1 + freq)) for term, freq in df.items()}
    return idf


# Example usage
documents = ["cat eats fish", "dog eats fish", "cat chases dog"]

idf_scores = compute_idf(documents)
print("IDF Scores:")
for term, score in idf_scores.items():
    print(term, ":", round(score, 3))
