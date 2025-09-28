from collections import Counter


def compute_tf(doc):
    """
    Compute term frequency for a single document.
    Returns dictionary: term -> normalized frequency
    """
    words = doc.lower().split()
    total_terms = len(words)
    counts = Counter(words)

    # Normalized TF
    tf = {term: count / total_terms for term, count in counts.items()}
    return tf


# Example
doc1 = "cat eats fish cat cat"
doc2 = "dog eats fish"

print("TF for doc1:", compute_tf(doc1))
print("TF for doc2:", compute_tf(doc2))
