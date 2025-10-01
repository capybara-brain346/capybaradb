from capybaradb.persistent_tfidf_index import PersistentTFIDFIndex


def main():
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Machine learning algorithms use statistical techniques to enable computers to improve their performance on tasks through experience.",
        "Cooking pasta is simple and delicious. First, boil water in a large pot and add salt. Then add the pasta and cook for eight to ten minutes. Drain the pasta and add your favorite sauce.",
        "Space exploration has advanced significantly in recent decades. NASA and private companies are developing new technologies for Mars missions. Rockets are becoming reusable, reducing the cost of space travel.",
        "Climate change poses significant challenges to our planet. Rising temperatures affect weather patterns and sea levels. Greenhouse gases from human activities trap heat in the atmosphere.",
        "Ancient civilizations built remarkable structures that still stand today. The pyramids of Egypt showcase advanced engineering knowledge. Roman aqueducts supplied water to cities across their empire.",
    ]

    print("=== Persistent TF-IDF Demo ===\n")

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

    print("=== Creating Persistent TF-IDF Index ===")
    index = PersistentTFIDFIndex("embeddings.sqlite", documents)
    
    print(f"Index Type: {index.index_type}")
    print(f"Document Count: {index.get_document_count()}")

    for query in test_queries:
        results = index.search(query, top_k=3)
        print(f"\nQuery: '{query}'")
        for rank, (doc_id, score) in enumerate(results, 1):
            doc_preview = index.get_document(doc_id)[:60]
            print(f"  {rank}. Doc {doc_id} (Score: {score:.4f}): {doc_preview}...")

    print("\n" + "=" * 50)

    

if __name__ == "__main__":
    main()
