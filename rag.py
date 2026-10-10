import sys
from pathlib import Path

from generation import generate_answer
from index import build_index
from ingest import load_and_split
from retrieval import retrieve


def main() -> int:
    # PDF paths are supplied as command-line arguments after rag.py.
    # Stop early if no input files were provided.
    if len(sys.argv) < 2:
        print("Usage: python rag.py file1.pdf [file2.pdf ...]")
        return 1

    # Expand '~' in user-provided paths and check each path before ingestion.
    pdf_paths = [Path(name).expanduser() for name in sys.argv[1:]]
    for path in pdf_paths:
        if not path.is_file() or path.suffix.lower() != ".pdf":
            print(f"PDF not found: {path}")
            return 1

    # Ingestion extracts PDF text, then splits it into passages. The index is
    # built once, before the question loop, so repeated questions reuse it.
    chunks = load_and_split(pdf_paths)
    if not chunks:
        print("No readable text was found in the PDFs.")
        return 1

    print(f"Loaded {len(chunks)} passages. Building the index...")
    store = build_index(chunks)
    print("Ready. Type 'exit' to quit.")

    # Keep accepting questions until the user types exit or closes the input.
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        # An empty line is ignored; no retrieval or model call is needed.
        if question.lower() == "exit":
            break
        if not question:
            continue

        # Retrieval selects the top passages for this question. Print a short
        # preview so the user can inspect what evidence the model will receive.
        documents = retrieve(store, question)
        print("\nRetrieved passages:")
        for number, doc in enumerate(documents, start=1):
            print(f"[{number}] {Path(doc.metadata['source']).name}, page {doc.metadata['page']}")
            print("   ", doc.page_content[:180].replace("\n", " "), "...")

        # Generation receives only the selected passages, not the full PDFs.
        print("\nAnswer:")
        print(generate_answer(question, documents))

    # A zero exit status means the program ended normally.
    return 0


if __name__ == "__main__":
    # Importing this module elsewhere should not start the interactive loop.
    raise SystemExit(main())
