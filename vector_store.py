"""Store document chunks in a persistent Chroma database."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from chunking import split_documents
from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)
from loaders import load_file


@dataclass(frozen=True) # "frozen=true" means that the results don't change after creation
class IndexResult:
    # Tell the command-line interface what happened to a file
    status: str
    chunks: int


def open_vector_store() -> Chroma:
    # Chroma creates the database on the first run and reopens it later
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(CHROMA_DIR),
    )


def _file_digest(path: Path) -> str:
    """Calculate a hash that changes when the file content changes."""
    digest = sha256()

    # Read the file in blocks (of 1 MiB( instead of loading it all into memory
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)

    # Return content as text
    return digest.hexdigest()


def index_file(store: Chroma, path: Path) -> IndexResult:
    """Add a new file or update a file that has changed."""
    path = path.expanduser().resolve()
    source = str(path)

    # The signature identifies both the file content and indexing settings.
    # Changing the file, model, or chunk sizes produces a new signature
    signature_text = (
        f"v1:{_file_digest(path)}:"
        f"{EMBEDDING_MODEL}:{CHUNK_SIZE}:{CHUNK_OVERLAP}"
    )
    signature = sha256(signature_text.encode()).hexdigest()

    # Find passages that were previously stored for this file
    existing = store.get(
        where={"source": source},
        include=["metadatas"],
    )
    old_ids = existing["ids"]
    old_metadata = existing["metadatas"] or []

    # Skip embedding when a complete version of this file already exists
    if (
        old_ids
        and len(old_metadata) == len(old_ids)
        and all(
            metadata
            and metadata.get("index_signature") == signature
            and metadata.get("chunk_count") == len(old_ids)
            for metadata in old_metadata
        )
    ):
        return IndexResult(status="unchanged", chunks=len(old_ids))

    # Load the file and split its text into chunks
    chunks = split_documents(load_file(path))

    if not chunks:
        # Remove old passages if the file no longer has readable text
        if old_ids:
            store.delete(ids=old_ids)

        return IndexResult(status="empty", chunks=0)

    # Use the file path to create a stable document identifier
    document_id = sha256(source.encode()).hexdigest()[:16]
    new_ids = []

    for number, chunk in enumerate(chunks, start=1):
        # Each chunk gets its own identifier inside Chroma
        chunk_id = sha256(
            f"{source}:{signature}:{number}".encode()
        ).hexdigest()

        # Keep the identifiers and signature with the passage
        chunk.metadata.update(
            document_id=document_id,
            chunk_id=chunk_id,
            chunk_count=len(chunks),
            index_signature=signature,
        )
        new_ids.append(chunk_id)

    # Store the new passages before removing the previous version
    store.add_documents(documents=chunks, ids=new_ids)

    # Delete passages from the previous version of this file
    stale_ids = list(set(old_ids) - set(new_ids))
    if stale_ids:
        store.delete(ids=stale_ids)

    return IndexResult(status="indexed", chunks=len(chunks))