"""Split loaded documents into smaller passages."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


def split_documents(documents: list[Document]) -> list[Document]:
    # Smaller passages are easier to embed and retrieve
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    # Each chunk keeps metadata such as its source file and PDF page
    return splitter.split_documents(documents)