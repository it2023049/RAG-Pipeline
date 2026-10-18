"""Read plain text and Markdown files."""

from pathlib import Path

from langchain_core.documents import Document


def load_text(path: Path) -> list[Document]:
    # Both TXT and Markdown files are read as UTF-8 text
    text = path.read_text(encoding="utf-8")

    # Do not create a document for an empty file
    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": str(path.resolve()),
                "file_type": path.suffix.lower().lstrip("."),
            },
        )
    ]