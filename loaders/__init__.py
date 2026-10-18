"""Choose a loader based on the file extension."""

from pathlib import Path

from langchain_core.documents import Document

from loaders.pdf import load_pdf
from loaders.text import load_text

# These are the file types that users can currently index
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def load_file(path: Path) -> list[Document]:
    # Use the extension to select the correct loader
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path)

    if suffix in {".txt", ".md"}:
        return load_text(path)

    raise ValueError(f"Unsupported file type: {path}")