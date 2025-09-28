import math


def dcg(relevances):
    """Compute DCG for a list of relevance scores"""
    return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg(ranked_relevances, k=None):
    """Compute nDCG@k"""
    if k:
        ranked_relevances = ranked_relevances[:k]
    dcg_val = dcg(ranked_relevances)
    ideal_relevances = sorted(ranked_relevances, reverse=True)
    idcg_val = dcg(ideal_relevances)
    return dcg_val / idcg_val if idcg_val > 0 else 0


# Example usage
# Relevance scores of retrieved documents
relevances = [3, 2, 3, 0, 1, 2]

print("nDCG@6:", round(ndcg(relevances, k=6), 3))
