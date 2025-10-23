### capybaradb

A lightweight vector database implementation built from scratch in Python. CapybaraDB provides semantic search capabilities using with support for document chunking, text extraction functions from multiple file formats, and flexible storage options.

---

### Features

- **Semantic Search**: sentence-transformers for accurate semantic similarity search
- **Document Chunking**: Optional token-based chunking using tiktoken for better search granularity
- **Multiple File Formats**: Support for PDF, DOCX, and TXT files with OCR capabilities
- **Flexible Storage**: In-memory or persistent storage with automatic serialization
- **GPU Support**: CUDA acceleration for faster embedding generation and search
- **Precision Control**: Support for different precision levels (float32, float16, binary)
- **Collection Management**: Organize documents into named collections

### Installation

Clone the repo
```bash
git clone https://github.com/capybara-brain346/capybaradb.git
```
```bash
cd capybaradb
```
Setup environment

**using venv**
```bash
python -m venv venv
source venv/bin/activate  # on macOS/Linux
venv\Scripts\activate     # on Windows
pip install -r requirements.txt
```

**using uv**
```bash
uv sync
```

## Quick Start

```python
from capybaradb.main import CapybaraDB

# Create a database with chunking enabled
db = CapybaraDB(collection="my_docs", chunking=True, chunk_size=512)

# Add documents
doc_id = db.add_document("Capybaras are the largest rodents in the world.")
db.add_document("They are semi-aquatic mammals native to South America.")

# Search for similar content
results = db.search("biggest rodent", top_k=3)
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['text']}")
    print(f"Document: {result['document']}")
    print("---")

# Save the database
db.save()
```

### File Processing

CapybaraDB can process various file formats:

```python
from capybaradb.utils import extract_text_from_file

# Extract text from different file types
text = extract_text_from_file("document.pdf")
text = extract_text_from_file("document.docx")
text = extract_text_from_file("document.txt")

# Add the extracted text to the database
doc_id = db.add_document(text)
```

### Configuration Options

```python
db = CapybaraDB(
    collection="my_collection",    # Collection name (optional)
    chunking=True,                 # Enable document chunking
    chunk_size=512,               # Chunk size in tokens
    precision="float32",          # Embedding precision: "float32", "float16", "binary"
    device="cuda"                 # Device: "cpu" or "cuda"
)
```

### API Reference

#### Constructor
- `collection` (str, optional): Name of the collection
- `chunking` (bool): Enable document chunking (default: False)
- `chunk_size` (int): Size of chunks in tokens (default: 512)
- `precision` (str): Embedding precision (default: "float32")
- `device` (str): Device for computation (default: "cpu")

#### Methods
- `add_document(text: str, doc_id: str = None) -> str`: Add a document to the database
- `search(query: str, top_k: int = 5) -> List[Dict]`: Search for similar documents
- `get_document(doc_id: str) -> str`: Retrieve a document by ID
- `save()`: Save the database to disk
- `load()`: Load the database from disk
- `clear()`: Clear all data from the database

### License

This project is licensed under the MIT License - see the LICENSE file for details.
