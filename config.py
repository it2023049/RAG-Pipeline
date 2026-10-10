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
