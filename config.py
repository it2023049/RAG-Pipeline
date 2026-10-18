"""Configuration shared by indexing and question answering."""

from hashlib import sha256
from pathlib import Path

# Ollama model used to turn each passage and question into comparable vectors
EMBEDDING_MODEL = "embeddinggemma"
# Ollama chat model used to write the final answer from retrieved passages
CHAT_MODEL = "gemma3:12b"

# Maximum number of characters in a passage created by the text splitter
CHUNK_SIZE = 1000
# Characters repeated between neighboring passages to preserve nearby context
CHUNK_OVERLAP = 200
# Maximum number of passages retrieved for each question (not number of PDFs)
TOP_K = 5

# Save the Chroma database inside the project
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"

# A different embedding model gets a different collection
# This prevents vectors from different models from being mixed
COLLECTION_NAME = (
    "rag_documents_"
    + sha256(EMBEDDING_MODEL.encode()).hexdigest()[:12]
)