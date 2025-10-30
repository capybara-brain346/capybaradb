# Building CapybaraDB: A Complete Vector Database Implementation

Welcome to the technical deep-dive into CapybaraDB, a lightweight yet powerful vector database built from scratch in Python. Today, we'll explore how this semantic search system works under the hood, from document ingestion to lightning-fast similarity search.

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Core Architecture](#core-architecture)
3. [The Embedding Model](#the-embedding-model)
4. [Document Processing Pipeline](#document-processing-pipeline)
5. [The Search Engine](#the-search-engine)
6. [Storage and Persistence](#storage-and-persistence)
7. [Advanced Features](#advanced-features)
8. [Putting It All Together](#putting-it-all-together)

## The Big Picture

CapybaraDB transforms raw text documents into a searchable semantic space. Here's how it works at a high level:

```
Documents → Chunking → Embedding → Vector Storage → Search
```

When you add a document, CapybaraDB:

1. **Ingests** the raw text (or extracts it from PDFs, DOCX files, etc.)
2. **Chunks** the text into manageable pieces (optional, for better granularity)
3. **Embeds** each chunk into a 384-dimensional semantic space
4. **Stores** the vectors alongside metadata
5. **Enables** semantic search by comparing query embeddings to stored vectors

The magic happens when you search: instead of traditional keyword matching, CapybaraDB finds documents based on *meaning* and *context*, not just exact words.

## Core Architecture

CapybaraDB is built with a clean, modular architecture:

### BaseIndex: The Foundation

At the heart of CapybaraDB is the `BaseIndex` class, a data structure that holds:

```python
class BaseIndex:
    documents: Dict[str, str]           # doc_id -> full document text
    chunks: Dict[str, Dict[str, str]]   # chunk_id -> {text, doc_id}
    vectors: Optional[torch.Tensor]     # All chunk embeddings
    chunk_ids: List[str]                # Order-preserving chunk IDs
    total_chunks: int
    total_documents: int
    embedding_dim: Optional[int]
```

This design keeps documents and their chunks separate while maintaining relationships through IDs. Why this separation? It allows us to:

- Return full documents when retrieving search results
- Track which chunk belongs to which document
- Maintain metadata without duplicating data

### Index: The Persistent Layer

The `Index` class extends `BaseIndex` with persistence:

```python
class Index(BaseIndex):
    def __init__(self, storage_path: Optional[Path] = None):
        super().__init__()
        self.storage = Storage(storage_path)
```

This is where the magic of auto-loading happens. When you create an `Index`, it immediately checks if a persisted version exists and loads it automatically. This design means you can stop and resume sessions without losing data.

### CapybaraDB: The Main Interface

The `CapybaraDB` class ties everything together:

```python
class CapybaraDB:
    def __init__(
        self,
        collection: Optional[str] = None,
        chunking: bool = False,
        chunk_size: int = 512,
        precision: Literal["binary", "float16", "float32"] = "float32",
        device: Literal["cpu", "cuda"] = "cpu",
    ):
```

Notice the flexibility: you can create multiple collections, control chunking, adjust precision, and choose your compute device.

## The Embedding Model

### Architecture

CapybaraDB uses `sentence-transformers/all-MiniLM-L6-v2`, a lightweight but powerful transformer model that converts text into 384-dimensional vectors.

```python
class EmbeddingModel:
    def __init__(
        self,
        precision: Literal["binary", "float16", "float32"] = "float32",
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)
```

The model is initialized once and reused for all embeddings, keeping operations fast and memory-efficient.

### The Embedding Process

When you call `embed()` on a document, here's what happens:

```python
def embed(self, documents: Union[str, List[str]]) -> torch.Tensor:
    encoded_documents = self.tokenizer(
        documents, padding=True, truncation=True, return_tensors="pt"
    )
    
    with torch.no_grad():
        model_output = self.model(**encoded_documents)
    
    sentence_embeddings = self._mean_pooling(
        model_output, encoded_documents["attention_mask"]
    )
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
```

**Step 1: Tokenization**

The text is split into tokens (words/subwords) that the model understands. Padding ensures all documents have the same length, truncation handles overly long inputs.

**Step 2: Forward Pass**

The model processes the tokens, producing context-aware representations for each token position.

**Step 3: Mean Pooling**

This is where the magic happens. Instead of using just the first token's embedding (like BERT's [CLS] token), we compute the average of all token embeddings, weighted by attention:

```python
def _mean_pooling(self, model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()
    return torch.sum(
        token_embeddings * input_mask_expanded, 1
    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
```

This captures the semantic content of the entire sentence, not just a single token.

**Step 4: Normalization**

L2 normalization makes all vectors unit-length. This is crucial because:

- It makes cosine similarity equivalent to dot product
- It stabilizes training and search
- It enables efficient binary embeddings

### Precision Modes

CapybaraDB supports three precision modes:

**Float32 (default)**: Full precision, highest accuracy

**Float16**: Half precision, ~50% memory savings, minimal accuracy loss

**Binary**: Each dimension becomes 0 or 1, 75% memory savings. The embedding process converts values > 0 to 1.0:

```python
if self.precision == "binary":
    sentence_embeddings = (sentence_embeddings > 0).float()
```

Binary embeddings use a scaled dot product during search to compensate for information loss.

## Document Processing Pipeline

### Adding a Document

When you add a document, here's the full journey:

```python
def add_document(self, text: str, doc_id: Optional[str] = None) -> str:
    if doc_id is None:
        doc_id = str(uuid.uuid4())
    
    self.index.documents[doc_id] = text
    self.index.total_documents += 1
```

**Step 1: ID Generation**

If no ID is provided, we generate a UUID. This ensures every document is uniquely identifiable.

**Step 2: Chunking (Optional)**

If chunking is enabled, the document is split using token-based chunking:

```python
if self.chunking:
    enc = tiktoken.get_encoding("cl100k_base")
    token_ids = enc.encode(text)
    chunks = []
    for i in range(0, len(token_ids), self.chunk_size):
        tok_chunk = token_ids[i : i + self.chunk_size]
        chunk_text = enc.decode(tok_chunk)
        chunks.append(chunk_text)
```

Why token-based chunking instead of character-based?

- Respects word boundaries
- Considers tokenizer structure
- Produces more semantically coherent chunks
- Works better with the embedding model

**Step 3: Create Chunks**

Each chunk gets its own UUID and is stored with metadata:

```python
for chunk in chunks:
    chunk_id = str(uuid.uuid4())
    self.index.chunks[chunk_id] = {"text": chunk, "doc_id": doc_id}
    chunk_ids.append(chunk_id)
    self.index.total_chunks += 1
```

**Step 4: Generate Embeddings**

All chunks are embedded in one batch:

```python
chunk_texts = [self.index.chunks[cid]["text"] for cid in chunk_ids]
chunk_embeddings = self.model.embed(chunk_texts)
```

Batch processing is key to performance. Embedding 100 chunks together is much faster than 100 individual embeddings.

**Step 5: Append to Vector Store**

This is where the vectors are added to the index:

```python
if self.index.vectors is None:
    self.index.vectors = chunk_embeddings
    self.index.chunk_ids = chunk_ids
    self.index.embedding_dim = chunk_embeddings.size(1)
else:
    self.index.vectors = torch.cat(
        [self.index.vectors, chunk_embeddings], dim=0
    )
    self.index.chunk_ids.extend(chunk_ids)
```

The first document creates the tensor. Subsequent documents are concatenated along the batch dimension.

**Step 6: Persistence**

If not in-memory mode, the index is saved immediately:

```python
if not self.index.storage.in_memory:
    self.index.save()
```

This means you can add documents and they're persisted incrementally—no manual save needed!

### File Support

CapybaraDB can extract text from multiple file formats:

**PDF** - Uses `pypdf` for text extraction, with optional OCR fallback:

```python
def extract_text_from_pdf(file_path, use_ocr=False):
    pdf_reader = pypdf.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
```

If text extraction fails or returns empty, CapybaraDB can fall back to OCR using TrOCR (Microsoft's transformer-based OCR model), converting PDF pages to images and extracting text.

**DOCX** - Uses `python-docx`:

```python
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
```

**TXT** - Direct file read with encoding handling

## The Search Engine

### The Search Process

Here's how search works end-to-end:

```python
def search(self, query: str, top_k: int = 5):
    if self.index.vectors is None:
        return []
    
    self.index.ensure_vectors_on_device(target_device)
    indices, scores = self.model.search(query, self.index.vectors, top_k)
    
    results = []
    for idx, score in zip(indices.tolist(), scores.tolist()):
        chunk_id = self.index.chunk_ids[idx]
        chunk_info = self.index.chunks[chunk_id]
        doc_id = chunk_info["doc_id"]
        
        results.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": chunk_info["text"],
            "score": score,
            "document": self.index.documents[doc_id],
        })
    
    return results
```

**Step 1: Query Embedding**

The query text is embedded using the same model:

```python
def search(self, query: str, embeddings: torch.Tensor, top_k: int):
    query_embedding = self.embed(query)
```

**Step 2: Similarity Computation**

The similarity between query and all stored vectors is computed:

```python
if self.precision == "binary":
    similarities = torch.matmul(
        embeddings.float(), 
        query_embedding.t().float()
    ) / query_embedding.size(1)
else:
    similarities = torch.matmul(embeddings, query_embedding.t())
```

Because embeddings are normalized (L2), this computes cosine similarity. For binary embeddings, we scale by the dimension count.

**Step 3: Top-K Selection**

We use `torch.topk` to find the most similar vectors:

```python
scores, indices = torch.topk(
    similarities.squeeze(),
    min(top_k, embeddings.size(0))
)
```

**Step 4: Result Assembly**

For each result, we reconstruct the full context by:
1. Looking up the chunk text
2. Finding the parent document ID
3. Retrieving the full document

This gives you both the specific chunk that matched and the full document context.

### Why This Design?

Notice that search returns chunks, not documents. Why?

- **Granularity**: A long document might only have one relevant paragraph
- **Precision**: You see exactly what matched, not entire irrelevant documents
- **Context**: You still get the full document through the `document` field

### Device Management

A subtle but important detail: CapybaraDB ensures vectors are on the correct device:

```python
def ensure_vectors_on_device(self, target_device: str):
    if self.vectors is not None and self.vectors.device.type != target_device:
        self.vectors = self.vectors.to(target_device)
```

This handles the common case where you load from disk (CPU) but want to search on GPU (CUDA).

## Storage and Persistence

### The Storage Layer

The `Storage` class handles persistence with NumPy's compressed NPZ format:

```python
def save(self, index) -> None:
    data = {
        "vectors": index.vectors.cpu().numpy(),
        "chunk_ids": np.array(index.chunk_ids),
        "chunk_texts": np.array([index.chunks[cid]["text"] for cid in index.chunk_ids]),
        "chunk_doc_ids": np.array([index.chunks[cid]["doc_id"] for cid in index.chunk_ids]),
        "doc_ids": np.array(list(index.documents.keys())),
        "doc_texts": np.array(list(index.documents.values())),
        "total_chunks": index.total_chunks,
        "total_documents": index.total_documents,
        "embedding_dim": index.embedding_dim or 0,
    }
    
    np.savez_compressed(self.file_path, **data)
```

Why NPZ?

- Compressed by default (saves space)
- Efficient binary format
- Handles large arrays well
- Cross-platform and language-agnostic

### In-Memory vs Persistent

CapybaraDB supports two modes:

**In-Memory**: No file path specified. Data stays in RAM, lost on exit.

**Persistent**: File path specified. Data is saved to disk after each `add_document()` call.

This dual-mode design enables both temporary experiments (in-memory) and production use (persistent).

## Advanced Features

### Collection Management

Each CapybaraDB instance can use a collection name:

```python
db = CapybaraDB(collection="my_docs")
```

This creates separate storage files for different collections, enabling you to organize multiple datasets independently.

### OCR Integration

When regular PDF text extraction fails, CapybaraDB can use OCR:

```python
def extract_text_from_pdf(file_path, use_ocr=False):
    if not text and use_ocr:
        images = convert_pdf_to_images(file_path, ocr_dpi)
        ocr_processor = OCRProcessor("microsoft/trocr-base-printed")
        text = ocr_processor.extract_text_from_images(images)
```

The OCR system uses TrOCR, a transformer-based model that's remarkably accurate for printed text.

### Logging System

CapybaraDB has a comprehensive logging system:

```python
def setup_logger(name: str = "CapybaraDB", level=logging.INFO):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
```

Every module gets its own logger, making debugging easy. The logs show exactly which operation happened in which file at which line.

## Putting It All Together

Let's trace through a complete example:

### Example 1: Simple Document Search

```python
from capybaradb.main import CapybaraDB

# Initialize
db = CapybaraDB(
    collection="research_papers",
    chunking=True,
    chunk_size=512,
    device="cuda"
)

# Add documents
doc1_id = db.add_document("Machine learning is transforming NLP...")
doc2_id = db.add_document("Deep neural networks excel at image recognition...")

# Search
results = db.search("artificial intelligence", top_k=2)

# Use results
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Matched text: {result['text'][:100]}...")
    print(f"Full document: {result['document']}")
    print("---")
```

What happens under the hood:

1. **Initialization**: 
   - Model loads (sentence-transformers/all-MiniLM-L6-v2)
   - Checks for existing `data/research_papers.npz`
   - If found, loads all vectors and metadata

2. **Adding doc1**:
   - Text: "Machine learning is transforming NLP..."
   - Chunking creates 1 chunk (text is short)
   - Embedding: `[384-dim vector]`
   - Stored in `index.vectors` (shape: 1×384)
   - Saved to disk

3. **Adding doc2**:
   - Text: "Deep neural networks excel at image recognition..."
   - Chunking creates 1 chunk
   - Embedding: `[384-dim vector]`
   - Concatenated with existing vectors (shape: 2×384)
   - Saved to disk

4. **Searching**:
   - Query embedded: `[384-dim vector]`
   - Compared to 2 stored vectors
   - Cosine similarities computed
   - Top 2 results returned
   - Each result includes chunk text, full document, and score

### Example 2: File Processing

```python
from capybaradb.utils import extract_text_from_file
from capybaradb.main import CapybaraDB

# Extract text from PDF
text = extract_text_from_file("research_paper.pdf", use_ocr=True)

# Add to database
db = CapybaraDB(chunking=True, chunk_size=256)
doc_id = db.add_document(text)
```

What happens:

1. **PDF extraction**: 
   - Attempts text extraction
   - If fails or returns empty, converts PDF to images
   - Uses TrOCR to extract text from images
   - Returns complete text

2. **Chunking**:
   - Large paper becomes multiple chunks of 256 tokens each
   - Each chunk gets unique ID and doc reference

3. **Embedding**:
   - All chunks embedded in single batch
   - Appended to vector store

4. **Persistence**:
   - Saved to disk automatically
   - Ready for future searches

## Design Decisions and Trade-offs

### Why Not Incremental HNSW?

CapybaraDB uses linear search through all vectors. For millions of documents, you'd want an approximate nearest neighbor (ANN) index like HNSW. But for most use cases (thousands to tens of thousands of documents), linear search is:

- **Simpler**: No index maintenance
- **Exact**: No approximation artifacts
- **Memory-efficient**: No additional index structures
- **Fast**: Vectorized operations are very fast on modern hardware

### Why Store Everything?

CapybaraDB stores full documents, chunks, and vectors. This redundancy is intentional:

- **Search quality**: You see both the matching snippet and full context
- **Retrieval**: Can fetch documents by ID without re-embedding
- **Flexibility**: Easy to extend with additional metadata

The trade-off is memory usage. For extremely large datasets, you'd want to store only references to disk.

### Why Token-Based Chunking?

Token-based chunking respects the embedding model's structure:

- Tokenizers understand subwords and special tokens
- Chunks align with model's text representation
- More likely to produce semantically coherent pieces

The alternative—character-based chunking—is simpler but produces worse results.

## The Bottom Line

CapybaraDB demonstrates that building a production-ready vector database doesn't require massive infrastructure. The core is surprisingly simple:

1. Embed text into vectors
2. Store vectors with metadata
3. Search using cosine similarity

The sophistication comes from:
- Careful attention to device management
- Flexible precision and chunking options
- Robust persistence mechanisms
- Clean abstractions for different use cases

Whether you're building a research prototype or a production system, CapybaraDB's architecture provides a solid foundation for semantic search applications.

## What's Next?

Possible enhancements:

- **ANN indices**: Add HNSW for million-scale datasets
- **Metadata filtering**: Filter results by document tags
- **Hybrid search**: Combine semantic with keyword matching
- **Multi-lingual**: Support for non-English text
- **Streaming**: Add documents without full re-indexing

The current implementation is complete and production-ready for moderate-scale applications. From here, the sky's the limit!
