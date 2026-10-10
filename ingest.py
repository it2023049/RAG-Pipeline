"""Read PDF files and split their text into passages."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_SIZE


def load_and_split(pdf_paths: list[Path]) -> list[Document]:
    # First extract text page by page. Keeping one Document per PDF page lets
    # later stages show exactly which file and page supplied a passage
    pages = []

    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)
        # Page numbers shown to readers start at 1; Python's indexes start at 0
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            # Image-only or blank pages may have no extractable text
            if text.strip():
                pages.append(
                    Document(
                        page_content=text,
                        # A full path identifies the PDF; page identifies its
                        # original page even after the text is split into chunks
                        metadata={"source": str(pdf_path.resolve()), "page": page_number},
                    )
                )

    # Split long pages into smaller passages for embedding and retrieval.
    # Overlap repeats nearby text so a fact at a boundary is less likely to
    # lose its surrounding context. The start index records chunk position.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    # split_documents copies each page's metadata to its resulting passages
    return splitter.split_documents(pages)
