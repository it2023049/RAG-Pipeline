"""Read PDF files and keep their page numbers."""

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(path: Path) -> list[Document]:
    pages = []

    # Read each PDF page separately so we can cite its page number later
    for page_number, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""

        # Scanned or empty pages may not contain extractable text
        if not text.strip():
            continue

        pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path.resolve()),
                    "file_type": "pdf",
                    "page": page_number,
                },
            )
        )

    return pages