from collections import defaultdict

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)

    def add_document(self, doc_id, text):
        for term in text.lower().split():
            self.index[term].add(doc_id)

    def search(self, query):
        terms = query.lower().split()
        if not terms:
            return set()

        result = self.index.get(terms[0], set()).copy()

        for term in terms[1:]:
            result &= self.index.get(term, set())

        return result

    def __repr__(self):
        return "\n".join(f"{term}: {sorted(list(docs))}" for term, docs in self.index.items())


docs = {
    1: "cat eats fish",
    2: "dog eats fish",
    3: "cat chases dog"
}

index = InvertedIndex()

for doc_id, text in docs.items():
    index.add_document(doc_id, text)

print("Inverted Index:")
print(index)

print("\nSearch results:")
print("cat AND fish:", index.search("cat fish"))
print("dog AND eats:", index.search("dog eats"))
print("chases:", index.search("chases"))
