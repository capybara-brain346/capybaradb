from capybaradb import BM25Index, TFIDFIndex, ContextualIndex
from capybaradb import SearchEngine


def main():
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Machine learning algorithms use statistical techniques to enable computers to improve their performance on tasks through experience.",
        "Cooking pasta is simple and delicious. First, boil water in a large pot and add salt. Then add the pasta and cook for eight to ten minutes. Drain the pasta and add your favorite sauce.",
        "Space exploration has advanced significantly in recent decades. NASA and private companies are developing new technologies for Mars missions. Rockets are becoming reusable, reducing the cost of space travel.",
        "Climate change poses significant challenges to our planet. Rising temperatures affect weather patterns and sea levels. Greenhouse gases from human activities trap heat in the atmosphere.",
        "Ancient civilizations built remarkable structures that still stand today. The pyramids of Egypt showcase advanced engineering knowledge. Roman aqueducts supplied water to cities across their empire.",
    ]

    print("=== CapybaraDB Vector Search Demo ===\n")

    print("Sample Documents:")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc[:80]}...")
    print()

    test_queries = [
        "machine learning algorithms",
        "ancient pyramids",
        "space missions",
        "cooking pasta",
    ]

    print("=== Testing TF-IDF Index ===")
    tfidf_index = TFIDFIndex(documents)
    tfidf_engine = SearchEngine(tfidf_index)

    print(f"Index Type: {tfidf_engine.get_index_type()}")
    print(f"Document Count: {tfidf_engine.get_document_count()}")

    for query in test_queries:
        results = tfidf_engine.search(query, top_k=3)
        print(f"\nQuery: '{query}'")
        for rank, (doc_id, score) in enumerate(results, 1):
            doc_preview = tfidf_engine.get_document(doc_id)[:60]
            print(f"  {rank}. Doc {doc_id} (Score: {score:.4f}): {doc_preview}...")

    print("\n" + "=" * 50)

    print("=== Testing BM25 Index ===")
    bm25_index = BM25Index(documents)
    bm25_engine = SearchEngine(bm25_index)

    print(f"Index Type: {bm25_engine.get_index_type()}")
    print(f"Document Count: {bm25_engine.get_document_count()}")

    for query in test_queries:
        results = bm25_engine.search(query, top_k=3)
        print(f"\nQuery: '{query}'")
        for rank, (doc_id, score) in enumerate(results, 1):
            doc_preview = bm25_engine.get_document(doc_id)[:60]
            print(f"  {rank}. Doc {doc_id} (Score: {score:.4f}): {doc_preview}...")

    print("\n" + "=" * 50)

    print("=== Testing Contextual Index ===")
    try:
        contextual_index = ContextualIndex(documents)
        contextual_engine = SearchEngine(contextual_index)

        print(f"Index Type: {contextual_engine.get_index_type()}")
        print(f"Document Count: {contextual_engine.get_document_count()}")

        for query in test_queries:
            results = contextual_engine.search(query, top_k=3)
            print(f"\nQuery: '{query}'")
            for rank, (doc_id, score) in enumerate(results, 1):
                doc_preview = contextual_engine.get_document(doc_id)[:60]
                print(f"  {rank}. Doc {doc_id} (Score: {score:.4f}): {doc_preview}...")

    except ImportError as e:
        print(f"Contextual Index unavailable due to missing dependencies: {e}")
        print("Install transformers and torch to use ContextualIndex")

    print("\n" + "=" * 50)

    print("=== Testing SearchEngine Features ===")
    engine = SearchEngine(tfidf_index)

    print("Adding new documents...")
    new_docs = [
        "Artificial intelligence and neural networks are transforming technology.",
        "Renewable energy sources like solar and wind power are becoming more efficient.",
    ]

    print(f"Documents before adding: {engine.get_document_count()}")
    engine.add_documents(new_docs)
    print(f"Documents after adding: {engine.get_document_count()}")

    print("\nSearching in expanded corpus:")
    results = engine.search("artificial intelligence", top_k=3)
    for rank, (doc_id, score) in enumerate(results, 1):
        doc_preview = engine.get_document(doc_id)[:60]
        print(f"  {rank}. Doc {doc_id} (Score: {score:.4f}): {doc_preview}...")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
