"""Create readable source labels for retrieved passages."""

from pathlib import Path

from langchain_core.documents import Document


def format_source(document: Document) -> str:
    # Show the file name instead of its full filesystem path
    source = document.metadata.get("source")
    label = Path(str(source)).name if source else "unknown source"

    # Only PDFs have page numbers in the current loaders
    page = document.metadata.get("page")
    if page is not None:
        return f"{label}, page {page}"

    return label