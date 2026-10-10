"""Embed passages and store their vectors in memory."""

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

from config import EMBEDDING_MODEL


def build_index(chunks: list[Document]) -> InMemoryVectorStore:
    # Ollama computes a vector for each passage using the configured model
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    # This store lives only in RAM; restarting the script rebuilds the index
    store = InMemoryVectorStore(embeddings)
    # Keep the Document text and metadata alongside each vector so retrieved
    # results still contain their source PDF and page number
    store.add_documents(chunks)
    return store
