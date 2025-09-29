from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


class ContextualEmbeddings:
    def __init__(self, docs) -> None:
        self.docs = docs
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def compute_embeddings(self):
        encoded_input = self._encode()

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self._mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        print("Sentence embeddings:")
        print(sentence_embeddings)

    def _encode(self):
        return self.tokenizer(
            self.docs, padding=True, truncation=True, return_tensors="pt"
        )

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


if __name__ == "__main__":
    documents = [
        """Machine learning is a subset of artificial intelligence that focuses on 
    building systems that learn from data. Machine learning algorithms use 
    statistical techniques to enable computers to improve their performance 
    on tasks through experience. Deep learning is a specialized form of 
    machine learning that uses neural networks with multiple layers.""",
        """Cooking pasta is simple and delicious. First, boil water in a large pot 
    and add salt. Then add the pasta and cook for eight to ten minutes. 
    Drain the pasta and add your favorite sauce. Italian cooking emphasizes 
    fresh ingredients and simple preparation techniques. Pasta dishes are 
    beloved around the world.""",
        """Space exploration has advanced significantly in recent decades. NASA and 
    private companies are developing new technologies for Mars missions. 
    Rockets are becoming reusable, reducing the cost of space travel. 
    Scientists study planets, stars, and galaxies to understand our universe. 
    The International Space Station orbits Earth as a research laboratory.""",
        """Climate change poses significant challenges to our planet. Rising 
    temperatures affect weather patterns and sea levels. Greenhouse gases 
    from human activities trap heat in the atmosphere. Scientists emphasize 
    the need for renewable energy and sustainable practices. Global 
    cooperation is essential to address climate change effectively.""",
        """Ancient civilizations built remarkable structures that still stand today. 
    The pyramids of Egypt showcase advanced engineering knowledge. Roman 
    aqueducts supplied water to cities across their empire. Ancient Greeks 
    made significant contributions to philosophy, mathematics, and democracy. 
    Archaeologists continue discovering artifacts that reveal secrets of 
    ancient cultures.""",
    ]
    contextual_embeddings = ContextualEmbeddings(documents)
    contextual_embeddings.compute_embeddings()
