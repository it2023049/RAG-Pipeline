"""Command-line interface for indexing files and asking questions."""

import argparse
from pathlib import Path

from citations import format_source
from generation import generate_answer
from loaders import SUPPORTED_SUFFIXES
from retrieval import retrieve
from vector_store import index_file, open_vector_store


def parse_arguments() -> argparse.Namespace:
    # Create two commands: one for indexing and one for questions
    parser = argparse.ArgumentParser(
        description="Local RAG with Ollama and Chroma"
    )
    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    indexing = commands.add_parser(
        "index",
        help="Add or update local files",
    )
    indexing.add_argument(
        "files",
        nargs="+",
        type=Path,
    )

    asking = commands.add_parser(
        "ask",
        help="Ask one question or enter interactive mode",
    )
    asking.add_argument(
        "question",
        nargs="?",
    )

    return parser.parse_args()


def run_index(files: list[Path]) -> int:
    # Convert user-provided paths into absolute paths
    paths = [
        path.expanduser().resolve()
        for path in files
    ]

    # Check every file before writing anything to Chroma
    for path in paths:
        if not path.is_file():
            print(f"File not found: {path}")
            return 1

        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            print(f"Unsupported file type: {path}")
            return 1

    store = open_vector_store()

    # Index each unique path only once
    for path in dict.fromkeys(paths):
        result = index_file(store, path)
        print(
            f"{path.name}: {result.status} "
            f"({result.chunks} passages)"
        )

    return 0


def print_answer(store, question: str) -> None:
    # Find relevant passages before calling the chat model
    documents = retrieve(store, question)

    # Show the evidence that will be given to the model
    print("\nRetrieved passages:")
    for number, document in enumerate(documents, start=1):
        print(f"[{number}] {format_source(document)}")

        preview = document.page_content[:180].replace("\n", " ")
        print(f"    {preview}...")

    print("\nAnswer:")
    print(generate_answer(question, documents))


def run_ask(question: str | None) -> int:
    store = open_vector_store()

    # Questions need at least one previously indexed passage
    if not store.get(limit=1)["ids"]:
        print("The index is empty. Run: python rag.py index file.pdf")
        return 1

    # Answer once if the question was provided in the command
    if question:
        print_answer(store, question)
        return 0

    # Otherwise, continue reading questions from the terminal
    print("Ready. Type 'exit' to quit.")

    while True:
        try:
            user_question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_question.lower() == "exit":
            break

        if user_question:
            print_answer(store, user_question)

    return 0


def main() -> int:
    args = parse_arguments()

    # Select the action requested by the user
    if args.command == "index":
        return run_index(args.files)

    return run_ask(args.question)


if __name__ == "__main__":
    raise SystemExit(main())