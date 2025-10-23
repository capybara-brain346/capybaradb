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
import torch

from capybaradb import CapybaraDB
from capybaradb.utils import extract_text_from_pdf


def demo() -> None:
    logging.basicConfig(level=logging.INFO)

    db = CapybaraDB(
        chunking=True,
        chunk_size=64,
        precision="binary",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Capybaras are large semi-aquatic rodents native to South America.",
        "PyTorch provides tensors and GPU acceleration for deep learning.",
    ]

    print("Adding documents...")
    input_text = extract_text_from_pdf(
        "./tests/data/Fine-Tuned LLM_SLM Use Cases in TrackML-Backend.pdf"
    )
    ids: list[str] = []
    doc_id = db.add_document(input_text)
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
            print(
                f"score={r['score']} | doc_id={r['doc_id']} | chunk_id={r['chunk_id']} | text={r['text']}"
            )


if __name__ == "__main__":
    demo()
