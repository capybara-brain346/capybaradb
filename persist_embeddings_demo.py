from capybaradb import ContextualIndex
from capybaradb.db import EmbeddingsDB
import numpy as np


def main():
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Machine learning algorithms use statistical techniques to enable computers to improve their performance on tasks through experience.",
        "Cooking pasta is simple and delicious. First, boil water in a large pot and add salt. Then add the pasta and cook for eight to ten minutes. Drain the pasta and add your favorite sauce.",
        "Space exploration has advanced significantly in recent decades. NASA and private companies are developing new technologies for Mars missions. Rockets are becoming reusable, reducing the cost of space travel.",
        "Climate change poses significant challenges to our planet. Rising temperatures affect weather patterns and sea levels. Greenhouse gases from human activities trap heat in the atmosphere.",
        "Ancient civilizations built remarkable structures that still stand today. The pyramids of Egypt showcase advanced engineering knowledge. Roman aqueducts supplied water to cities across their empire.",
    ]

    try:
        index = ContextualIndex(documents)
        index.compute_index()
    except ImportError as e:
        print(f"Contextual Index unavailable due to missing dependencies: {e}")
        print("Install transformers and torch to use ContextualIndex")
        return

    db = EmbeddingsDB("embeddings.sqlite")
    db.initialize()

    for doc_id in range(len(documents)):
        vec = index.embeddings[doc_id].cpu().numpy().astype(np.float32)
        db.upsert(doc_id, vec)

    print(f"Saved embeddings: {db.count()}")

    loaded = db.get(0)
    print(f"Loaded dim for doc 0: {None if loaded is None else loaded.shape[0]}")

    query = "machine learning algorithms"
    query_encoded = index.tokenizer([query], padding=True, truncation=True, return_tensors="pt")
    import torch
    with torch.no_grad():
        q_out = index.model(**query_encoded)
    q_emb = index._mean_pooling(q_out, query_encoded["attention_mask"]).squeeze(0)
    q_vec = torch.nn.functional.normalize(q_emb, p=2, dim=0).cpu().numpy().astype(np.float32)

    top = []
    for batch in db.iterate(batch_size=1024):
        ids = [doc_id for doc_id, _ in batch]
        mat = np.stack([vec for _, vec in batch], axis=0)
        scores = mat @ q_vec
        for i, s in zip(ids, scores):
            top.append((i, float(s)))

    top.sort(key=lambda x: x[1], reverse=True)
    top_k = top[:3]
    print("Top results:")
    for rank, (doc_id, score) in enumerate(top_k, 1):
        preview = documents[doc_id][:60]
        print(f"  {rank}. Doc {doc_id} (Score: {score:.4f}): {preview}...")


if __name__ == "__main__":
    main()


