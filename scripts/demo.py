from capybaradb.main import CapybaraDB
from capybaradb.utils import extract_text_from_file


def main():
    db = CapybaraDB(collection="demo", chunking=True, chunk_size=256)

    sample_texts = [
        "Capybaras are the largest rodents in the world. They are semi-aquatic mammals.",
        "The capybara's scientific name is Hydrochoerus hydrochaeris. They are native to South America.",
        "Capybaras are highly social animals and can live in groups of 10-20 individuals.",
    ]

    print("Adding documents to the database...")
    doc_ids = []
    for text in sample_texts:
        doc_id = db.add_document(text)
        doc_ids.append(doc_id)
        print(f"Added document with ID: {doc_id}")

    print("\nPerforming searches:")
    queries = ["largest rodent", "social behavior", "scientific classification"]

    for query in queries:
        print(f"\nSearch query: '{query}'")
        results = db.search(query, top_k=2)

        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['score']:.4f}):")
            print(f"Text: {result['text']}")

    print("\nTrying to read and index a text file...")
    try:
        text = extract_text_from_file(
            "./tests/data/CNN-Based Classifiers and Fine-Tune.txt"
        )
        doc_id = db.add_document(text)
        print(f"Successfully added text file with ID: {doc_id}")

        results = db.search("test content", top_k=1)
        if results:
            print("\nFound matching content from file:")
            print(f"Score: {results[0]['score']:.4f}")
            print(f"Text: {results[0]['text']}")
    except Exception as e:
        print(f"Error processing file: {e}")

    db.save()
    print("\nDatabase has been saved to disk.")


if __name__ == "__main__":
    main()
