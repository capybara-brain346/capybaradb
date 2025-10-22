"""Example usage for capybaradb.CapybaraDB.

Demonstrates:
- creating the DB with different precisions and chunking
- adding documents
- searching
- retrieving documents

Run this file directly for a quick demo (requires model & tiktoken for chunking).
"""

from time import sleep
import logging

from capybaradb import CapybaraDB


def demo() -> None:
    logging.basicConfig(level=logging.INFO)

    db = CapybaraDB(chunking=True, chunk_size=64, precision="float32", device="cpu")

    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Capybaras are large semi-aquatic rodents native to South America.",
        "PyTorch provides tensors and GPU acceleration for deep learning.",
    ]

    print("Adding documents...")
    ids: list[str] = []
    for d in docs:
        doc_id = db.add_document(d)
        print(f"added doc {doc_id[:8]}...")
        ids.append(doc_id)

    sleep(0.1)

    queries = [
        "rodent native to South America",
        "fast fox",
        "GPU accelerated tensor library",
    ]

    for q in queries:
        print("\nQuery:", q)
        results = db.search(q, top_k=3)
        for r in results:
            print(f"score={r['score']:.4f} doc_id={r['doc_id'][:8]} text={r['text']}")

    print("\nRetrieving full document:")
    print(db.get_document(ids[1]))


if __name__ == "__main__":
    demo()
